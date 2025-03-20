import json
import logging
import os
import sys
import threading
import warnings
from contextlib import contextmanager
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    TypedDict,
    Union,
    cast,
)

from dotenv import load_dotenv
from pydantic import BaseModel

from crewai.utilities.events.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMCallType,
    LLMStreamChunkEvent,
)
from crewai.utilities.events.tool_usage_events import ToolExecutionErrorEvent

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    import litellm
    from litellm import Choices
    from litellm.litellm_core_utils.get_supported_openai_params import (
        get_supported_openai_params,
    )
    from litellm.types.utils import ModelResponse
    from litellm.utils import supports_response_schema


from crewai.utilities.events import crewai_event_bus
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
    # mistral
    "mistral-tiny": 32768,
    "mistral-small-latest": 32768,
    "mistral-medium-latest": 32768,
    "mistral-large-latest": 32768,
    "mistral-large-2407": 32768,
    "mistral-large-2402": 32768,
    "mistral/mistral-tiny": 32768,
    "mistral/mistral-small-latest": 32768,
    "mistral/mistral-medium-latest": 32768,
    "mistral/mistral-large-latest": 32768,
    "mistral/mistral-large-2407": 32768,
    "mistral/mistral-large-2402": 32768,
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


class Delta(TypedDict):
    content: Optional[str]
    role: Optional[str]


class StreamingChoices(TypedDict):
    delta: Delta
    index: int
    finish_reason: Optional[str]


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
        stream: bool = False,
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
        self.stream = stream

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
        ANTHROPIC_PREFIXES = ("anthropic/", "claude-", "claude/")
        return any(prefix in model.lower() for prefix in ANTHROPIC_PREFIXES)

    def _prepare_completion_params(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """Prepare parameters for the completion call.

        Args:
            messages: Input messages for the LLM
            tools: Optional list of tool schemas
            callbacks: Optional list of callback functions
            available_functions: Optional dict of available functions

        Returns:
            Dict[str, Any]: Parameters for the completion call
        """
        # --- 1) Format messages according to provider requirements
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
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
            "stream": self.stream,
            "tools": tools,
            "reasoning_effort": self.reasoning_effort,
            **self.additional_params,
        }

        # Remove None values from params
        return {k: v for k, v in params.items() if v is not None}

    def _handle_streaming_response(
        self,
        params: Dict[str, Any],
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Handle a streaming response from the LLM.

        Args:
            params: Parameters for the completion call
            callbacks: Optional list of callback functions
            available_functions: Dict of available functions

        Returns:
            str: The complete response text

        Raises:
            Exception: If no content is received from the streaming response
        """
        # --- 1) Initialize response tracking
        full_response = ""
        last_chunk = None
        chunk_count = 0
        usage_info = None

        # --- 2) Make sure stream is set to True and include usage metrics
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}

        try:
            # --- 3) Process each chunk in the stream
            for chunk in litellm.completion(**params):
                chunk_count += 1
                last_chunk = chunk

                # Extract content from the chunk
                chunk_content = None

                # Safely extract content from various chunk formats
                try:
                    # Try to access choices safely
                    choices = None
                    if isinstance(chunk, dict) and "choices" in chunk:
                        choices = chunk["choices"]
                    elif hasattr(chunk, "choices"):
                        # Check if choices is not a type but an actual attribute with value
                        if not isinstance(getattr(chunk, "choices"), type):
                            choices = getattr(chunk, "choices")

                    # Try to extract usage information if available
                    if isinstance(chunk, dict) and "usage" in chunk:
                        usage_info = chunk["usage"]
                    elif hasattr(chunk, "usage"):
                        # Check if usage is not a type but an actual attribute with value
                        if not isinstance(getattr(chunk, "usage"), type):
                            usage_info = getattr(chunk, "usage")

                    if choices and len(choices) > 0:
                        choice = choices[0]

                        # Handle different delta formats
                        delta = None
                        if isinstance(choice, dict) and "delta" in choice:
                            delta = choice["delta"]
                        elif hasattr(choice, "delta"):
                            delta = getattr(choice, "delta")

                        # Extract content from delta
                        if delta:
                            # Handle dict format
                            if isinstance(delta, dict):
                                if "content" in delta and delta["content"] is not None:
                                    chunk_content = delta["content"]
                            # Handle object format
                            elif hasattr(delta, "content"):
                                chunk_content = getattr(delta, "content")

                            # Handle case where content might be None or empty
                            if chunk_content is None and isinstance(delta, dict):
                                # Some models might send empty content chunks
                                chunk_content = ""
                except Exception as e:
                    logging.debug(f"Error extracting content from chunk: {e}")
                    logging.debug(f"Chunk format: {type(chunk)}, content: {chunk}")

                # Only add non-None content to the response
                if chunk_content is not None:
                    # Add the chunk content to the full response
                    full_response += chunk_content

                    # Emit the chunk event
                    crewai_event_bus.emit(
                        self,
                        event=LLMStreamChunkEvent(chunk=chunk_content),
                    )

            # --- 4) Fallback to non-streaming if no content received
            if not full_response.strip() and chunk_count == 0:
                logging.warning(
                    "No chunks received in streaming response, falling back to non-streaming"
                )
                non_streaming_params = params.copy()
                non_streaming_params["stream"] = False
                non_streaming_params.pop(
                    "stream_options", None
                )  # Remove stream_options for non-streaming call
                return self._handle_non_streaming_response(
                    non_streaming_params, callbacks, available_functions
                )

            # --- 5) Handle empty response with chunks
            if not full_response.strip() and chunk_count > 0:
                logging.warning(
                    f"Received {chunk_count} chunks but no content was extracted"
                )
                if last_chunk is not None:
                    try:
                        # Try to extract content from the last chunk's message
                        choices = None
                        if isinstance(last_chunk, dict) and "choices" in last_chunk:
                            choices = last_chunk["choices"]
                        elif hasattr(last_chunk, "choices"):
                            if not isinstance(getattr(last_chunk, "choices"), type):
                                choices = getattr(last_chunk, "choices")

                        if choices and len(choices) > 0:
                            choice = choices[0]

                            # Try to get content from message
                            message = None
                            if isinstance(choice, dict) and "message" in choice:
                                message = choice["message"]
                            elif hasattr(choice, "message"):
                                message = getattr(choice, "message")

                            if message:
                                content = None
                                if isinstance(message, dict) and "content" in message:
                                    content = message["content"]
                                elif hasattr(message, "content"):
                                    content = getattr(message, "content")

                                if content:
                                    full_response = content
                                    logging.info(
                                        f"Extracted content from last chunk message: {full_response}"
                                    )
                    except Exception as e:
                        logging.debug(f"Error extracting content from last chunk: {e}")
                        logging.debug(
                            f"Last chunk format: {type(last_chunk)}, content: {last_chunk}"
                        )

            # --- 6) If still empty, raise an error instead of using a default response
            if not full_response.strip():
                raise Exception(
                    "No content received from streaming response. Received empty chunks or failed to extract content."
                )

            # --- 7) Check for tool calls in the final response
            tool_calls = None
            try:
                if last_chunk:
                    choices = None
                    if isinstance(last_chunk, dict) and "choices" in last_chunk:
                        choices = last_chunk["choices"]
                    elif hasattr(last_chunk, "choices"):
                        if not isinstance(getattr(last_chunk, "choices"), type):
                            choices = getattr(last_chunk, "choices")

                    if choices and len(choices) > 0:
                        choice = choices[0]

                        message = None
                        if isinstance(choice, dict) and "message" in choice:
                            message = choice["message"]
                        elif hasattr(choice, "message"):
                            message = getattr(choice, "message")

                        if message:
                            if isinstance(message, dict) and "tool_calls" in message:
                                tool_calls = message["tool_calls"]
                            elif hasattr(message, "tool_calls"):
                                tool_calls = getattr(message, "tool_calls")
            except Exception as e:
                logging.debug(f"Error checking for tool calls: {e}")

            # --- 8) If no tool calls or no available functions, return the text response directly
            if not tool_calls or not available_functions:
                # Log token usage if available in streaming mode
                self._handle_streaming_callbacks(callbacks, usage_info, last_chunk)
                # Emit completion event and return response
                self._handle_emit_call_events(full_response, LLMCallType.LLM_CALL)
                return full_response

            # --- 9) Handle tool calls if present
            tool_result = self._handle_tool_call(tool_calls, available_functions)
            if tool_result is not None:
                return tool_result

            # --- 10) Log token usage if available in streaming mode
            self._handle_streaming_callbacks(callbacks, usage_info, last_chunk)

            # --- 11) Emit completion event and return response
            self._handle_emit_call_events(full_response, LLMCallType.LLM_CALL)
            return full_response

        except Exception as e:
            logging.error(f"Error in streaming response: {str(e)}")
            if full_response.strip():
                logging.warning(f"Returning partial response despite error: {str(e)}")
                self._handle_emit_call_events(full_response, LLMCallType.LLM_CALL)
                return full_response

            # Emit failed event and re-raise the exception
            crewai_event_bus.emit(
                self,
                event=LLMCallFailedEvent(error=str(e)),
            )
            raise Exception(f"Failed to get streaming response: {str(e)}")

    def _handle_streaming_callbacks(
        self,
        callbacks: Optional[List[Any]],
        usage_info: Optional[Dict[str, Any]],
        last_chunk: Optional[Any],
    ) -> None:
        """Handle callbacks with usage info for streaming responses.

        Args:
            callbacks: Optional list of callback functions
            usage_info: Usage information collected during streaming
            last_chunk: The last chunk received from the streaming response
        """
        if callbacks and len(callbacks) > 0:
            for callback in callbacks:
                if hasattr(callback, "log_success_event"):
                    # Use the usage_info we've been tracking
                    if not usage_info:
                        # Try to get usage from the last chunk if we haven't already
                        try:
                            if last_chunk:
                                if (
                                    isinstance(last_chunk, dict)
                                    and "usage" in last_chunk
                                ):
                                    usage_info = last_chunk["usage"]
                                elif hasattr(last_chunk, "usage"):
                                    if not isinstance(
                                        getattr(last_chunk, "usage"), type
                                    ):
                                        usage_info = getattr(last_chunk, "usage")
                        except Exception as e:
                            logging.debug(f"Error extracting usage info: {e}")

                    if usage_info:
                        callback.log_success_event(
                            kwargs={},  # We don't have the original params here
                            response_obj={"usage": usage_info},
                            start_time=0,
                            end_time=0,
                        )

    def _handle_non_streaming_response(
        self,
        params: Dict[str, Any],
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Handle a non-streaming response from the LLM.

        Args:
            params: Parameters for the completion call
            callbacks: Optional list of callback functions
            available_functions: Dict of available functions

        Returns:
            str: The response text
        """
        # --- 1) Make the completion call
        response = litellm.completion(**params)

        # --- 2) Extract response message and content
        response_message = cast(Choices, cast(ModelResponse, response).choices)[
            0
        ].message
        text_response = response_message.content or ""

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

        # --- 4) Check for tool calls
        tool_calls = getattr(response_message, "tool_calls", [])

        # --- 5) If no tool calls or no available functions, return the text response directly
        if not tool_calls or not available_functions:
            self._handle_emit_call_events(text_response, LLMCallType.LLM_CALL)
            return text_response

        # --- 6) Handle tool calls if present
        tool_result = self._handle_tool_call(tool_calls, available_functions)
        if tool_result is not None:
            return tool_result

        # --- 7) If tool call handling didn't return a result, emit completion event and return text response
        self._handle_emit_call_events(text_response, LLMCallType.LLM_CALL)
        return text_response

    def _handle_tool_call(
        self,
        tool_calls: List[Any],
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Handle a tool call from the LLM.

        Args:
            tool_calls: List of tool calls from the LLM
            available_functions: Dict of available functions

        Returns:
            Optional[str]: The result of the tool call, or None if no tool call was made
        """
        # --- 1) Validate tool calls and available functions
        if not tool_calls or not available_functions:
            return None

        # --- 2) Extract function name from first tool call
        tool_call = tool_calls[0]
        function_name = tool_call.function.name
        function_args = {}  # Initialize to empty dict to avoid unbound variable

        # --- 3) Check if function is available
        if function_name in available_functions:
            try:
                # --- 3.1) Parse function arguments
                function_args = json.loads(tool_call.function.arguments)
                fn = available_functions[function_name]

                # --- 3.2) Execute function
                result = fn(**function_args)

                # --- 3.3) Emit success event
                self._handle_emit_call_events(result, LLMCallType.TOOL_CALL)
                return result
            except Exception as e:
                # --- 3.4) Handle execution errors
                fn = available_functions.get(
                    function_name, lambda: None
                )  # Ensure fn is always a callable
                logging.error(f"Error executing function '{function_name}': {e}")
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
                    event=LLMCallFailedEvent(error=f"Tool execution error: {str(e)}"),
                )
        return None

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
        """
        # --- 1) Emit call started event
        crewai_event_bus.emit(
            self,
            event=LLMCallStartedEvent(
                messages=messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
            ),
        )

        # --- 2) Validate parameters before proceeding with the call
        self._validate_call_params()

        # --- 3) Convert string messages to proper format if needed
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # --- 4) Handle O1 model special case (system messages not supported)
        if "o1" in self.model.lower():
            for message in messages:
                if message.get("role") == "system":
                    message["role"] = "assistant"

        # --- 5) Set up callbacks if provided
        with suppress_warnings():
            if callbacks and len(callbacks) > 0:
                self.set_callbacks(callbacks)

            try:
                # --- 6) Prepare parameters for the completion call
                params = self._prepare_completion_params(messages, tools)

                # --- 7) Make the completion call and handle response
                if self.stream:
                    return self._handle_streaming_response(
                        params, callbacks, available_functions
                    )
                else:
                    return self._handle_non_streaming_response(
                        params, callbacks, available_functions
                    )

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

        # Handle O1 models specially
        if "o1" in self.model.lower():
            formatted_messages = []
            for msg in messages:
                # Convert system messages to assistant messages
                if msg["role"] == "system":
                    formatted_messages.append(
                        {"role": "assistant", "content": msg["content"]}
                    )
                else:
                    formatted_messages.append(msg)
            return formatted_messages

        # Handle Mistral models - they require the last message to have a role of 'user' or 'tool'
        if "mistral" in self.model.lower():
            # Check if the last message has a role of 'assistant'
            if messages and messages[-1]["role"] == "assistant":
                # Add a dummy user message to ensure the last message has a role of 'user'
                messages = (
                    messages.copy()
                )  # Create a copy to avoid modifying the original
                messages.append({"role": "user", "content": "Please continue."})
            return messages

        # Handle Anthropic models
        if not self.is_anthropic:
            return messages

        # Anthropic requires messages to start with 'user' role
        if not messages or messages[0]["role"] == "system":
            # If first message is system or empty, add a placeholder user message
            return [{"role": "user", "content": "."}, *messages]

        return messages

    def _get_custom_llm_provider(self) -> Optional[str]:
        """
        Derives the custom_llm_provider from the model string.
        - For example, if the model is "openrouter/deepseek/deepseek-chat", returns "openrouter".
        - If the model is "gemini/gemini-1.5-pro", returns "gemini".
        - If there is no '/', defaults to "openai".
        """
        if "/" in self.model:
            return self.model.split("/")[0]
        return None

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
            provider = self._get_custom_llm_provider()
            return litellm.utils.supports_function_calling(
                self.model, custom_llm_provider=provider
            )
        except Exception as e:
            logging.error(f"Failed to check function calling support: {str(e)}")
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
