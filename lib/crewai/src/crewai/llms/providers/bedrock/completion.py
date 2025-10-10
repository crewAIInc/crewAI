import json
import logging
import os
from typing import Any

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)


try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    raise ImportError(
        "AWS Bedrock native provider not available, to install: `uv add boto3`"
    ) from None


class BedrockCompletion(BaseLLM):
    """AWS Bedrock native completion implementation.

    This class provides direct integration with AWS Bedrock,
    supporting both Text Completion and Messages APIs for Anthropic Claude models,
    as well as other Bedrock-supported models.
    """

    def __init__(
        self,
        model: str = "anthropic.claude-3-7-sonnet-20250219-v1:0",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        region_name: str = "us-east-1",
        temperature: float | None = None,
        max_tokens: int = 4096,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: list[str] | None = None,
        stream: bool = False,
        api_type: str = "messages",  # "messages" or "text_completion"
        **kwargs,
    ):
        """Initialize AWS Bedrock completion client.

        Args:
            model: Bedrock model ID (e.g., 'anthropic.claude-3-5-sonnet-20241022-v2:0')
            aws_access_key_id: AWS access key ID (defaults to AWS_ACCESS_KEY_ID env var)
            aws_secret_access_key: AWS secret access key (defaults to AWS_SECRET_ACCESS_KEY env var)
            aws_session_token: AWS session token (defaults to AWS_SESSION_TOKEN env var)
            region_name: AWS region name (defaults to us-east-1)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter (for Claude models)
            stop_sequences: Stop sequences
            stream: Enable streaming responses
            api_type: API type to use ("messages" or "text_completion")
            **kwargs: Additional parameters
        """

        super().__init__(
            model=model,
            temperature=temperature,
            stop=stop_sequences or [],
            **kwargs,
        )

        # Initialize Bedrock client
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret_access_key
            or os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=aws_session_token or os.getenv("AWS_SESSION_TOKEN"),
            region_name=region_name,
        )

        self.client = session.client("bedrock-runtime")
        self.region_name = region_name

        # Store completion parameters
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.stream = stream
        self.stop_sequences = stop_sequences or []
        self.api_type = api_type

        # Model-specific settings
        self.is_claude_model = "claude" in model.lower()
        self.is_claude_3_plus = any(
            v in model.lower() for v in ["claude-3", "claude-3-5"]
        )
        self.supports_tools = self.is_claude_3_plus and api_type == "messages"
        self.supports_streaming = True

        # Handle inference profiles for newer Claude models
        self.model_id = self._get_model_or_inference_profile(model)

        # Determine max tokens based on model if not specified
        if self.max_tokens == 4096 and self.is_claude_model:
            # Claude models can handle more tokens
            self.max_tokens = 4000  # Stay within Bedrock limits

    def call(
        self,
        messages: str | list[dict[str, str]],
        tools: list[dict] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str | Any:
        """Call AWS Bedrock API.

        Args:
            messages: Input messages for the completion
            tools: List of tool/function definitions
            callbacks: Callback functions (not used in native implementation)
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call

        Returns:
            Completion response or tool call result
        """
        try:
            # Emit call started event
            self._emit_call_started_event(
                messages=messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
            )

            # Choose API based on model and configuration
            if self.api_type == "text_completion" or not self.is_claude_3_plus:
                return self._handle_text_completion_api(
                    messages, tools, available_functions, from_task, from_agent
                )
            return self._handle_messages_api(
                messages, tools, available_functions, from_task, from_agent
            )

        except Exception as e:
            error_msg = f"AWS Bedrock API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    def _handle_text_completion_api(
        self,
        messages: str | list[dict[str, str]],
        tools: list[dict] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str:
        """Handle Text Completion API for Claude models."""
        # Format messages for text completion
        prompt = self._format_messages_for_text_completion(messages)

        # Prepare request body
        body = {
            "prompt": prompt,
            "max_tokens_to_sample": self.max_tokens,
        }

        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.top_k is not None:
            body["top_k"] = self.top_k
        if self.stop_sequences:
            body["stop_sequences"] = self.stop_sequences

        try:
            if self.stream:
                return self._handle_streaming_text_completion(
                    body, available_functions, from_task, from_agent
                )
            return self._handle_text_completion(
                body, available_functions, from_task, from_agent
            )
        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e
            raise

    def _handle_messages_api(
        self,
        messages: str | list[dict[str, str]],
        tools: list[dict] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str | Any:
        """Handle Messages API for Claude 3+ models."""
        # Format messages for Messages API
        formatted_messages, system_message = self._format_messages_for_messages_api(
            messages
        )

        # Prepare request body
        body = {
            "messages": formatted_messages,
            "max_tokens": self.max_tokens,
            "anthropic_version": "bedrock-2023-05-31",
        }

        if system_message:
            body["system"] = system_message
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.top_k is not None:
            body["top_k"] = self.top_k
        if self.stop_sequences:
            body["stop_sequences"] = self.stop_sequences

        # Add tools if provided and supported
        if tools and self.supports_tools:
            body["tools"] = self._convert_tools_for_interference(tools)

        try:
            if self.stream:
                return self._handle_streaming_messages(
                    body, available_functions, from_task, from_agent
                )
            return self._handle_messages_completion(
                body, available_functions, from_task, from_agent
            )
        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e
            raise

    def _handle_text_completion(
        self,
        body: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str:
        """Handle non-streaming text completion."""
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response["body"].read())
            content = response_body.get("completion", "")

            # Apply stop words
            content = self._apply_stop_words(content)

            # Track token usage if available
            if "usage" in response_body:
                self._track_token_usage_internal(response_body["usage"])

            self._emit_call_completed_event(
                response=content,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=body.get("prompt", ""),
            )

            return content

        except (ClientError, BotoCoreError) as e:
            error_msg = f"Bedrock text completion failed: {e}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _handle_messages_completion(
        self,
        body: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str | Any:
        """Handle non-streaming messages completion."""
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response["body"].read())

            # Track token usage if available
            if "usage" in response_body:
                self._track_token_usage_internal(response_body["usage"])

            # Handle tool use
            if "content" in response_body and available_functions:
                for content_block in response_body["content"]:
                    if content_block.get("type") == "tool_use":
                        function_name = content_block.get("name")
                        function_args = content_block.get("input", {})

                        result = self._handle_tool_execution(
                            function_name=function_name,
                            function_args=function_args,
                            available_functions=available_functions,
                            from_task=from_task,
                            from_agent=from_agent,
                        )

                        if result is not None:
                            return result

            # Extract text content
            content = ""
            if "content" in response_body:
                for content_block in response_body["content"]:
                    if content_block.get("type") == "text":
                        content += content_block.get("text", "")

            content = self._apply_stop_words(content)

            self._emit_call_completed_event(
                response=content,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=body.get("messages", []),
            )

            return content

        except (ClientError, BotoCoreError) as e:
            error_msg = f"Bedrock messages completion failed: {e}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _handle_streaming_text_completion(
        self,
        body: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str:
        """Handle streaming text completion."""
        full_response = ""

        try:
            response = self.client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json",
            )

            stream = response.get("body")
            if stream:
                for event in stream:
                    chunk = event.get("chunk")
                    if chunk:
                        chunk_data = json.loads(chunk.get("bytes").decode())
                        if "completion" in chunk_data:
                            text_chunk = chunk_data["completion"]
                            full_response += text_chunk
                            self._emit_stream_chunk_event(
                                chunk=text_chunk,
                                from_task=from_task,
                                from_agent=from_agent,
                            )

        except (ClientError, BotoCoreError) as e:
            error_msg = f"Bedrock streaming text completion failed: {e}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Apply stop words to full response
        full_response = self._apply_stop_words(full_response)

        # Emit completion event
        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=body.get("prompt", ""),
        )

        return full_response

    def _handle_streaming_messages(
        self,
        body: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str:
        """Handle streaming messages completion."""
        full_response = ""

        try:
            response = self.client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json",
            )

            stream = response.get("body")
            if stream:
                for event in stream:
                    chunk = event.get("chunk")
                    if chunk:
                        chunk_data = json.loads(chunk.get("bytes").decode())

                        # Handle content delta
                        if "delta" in chunk_data and "text" in chunk_data["delta"]:
                            text_chunk = chunk_data["delta"]["text"]
                            full_response += text_chunk
                            self._emit_stream_chunk_event(
                                chunk=text_chunk,
                                from_task=from_task,
                                from_agent=from_agent,
                            )

                        # Handle tool use streaming (if supported)
                        elif "content_block_delta" in chunk_data:
                            delta = chunk_data["content_block_delta"]
                            if delta.get("type") == "tool_use":
                                # Handle tool use streaming
                                pass  # Implementation depends on Bedrock streaming format

        except (ClientError, BotoCoreError) as e:
            error_msg = f"Bedrock streaming messages failed: {e}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Apply stop words to full response
        full_response = self._apply_stop_words(full_response)

        # Emit completion event
        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=body.get("messages", []),
        )

        return full_response

    def _format_messages_for_text_completion(
        self, messages: str | list[dict[str, str]]
    ) -> str:
        """Format messages for Claude Text Completion API.

        Text Completion API expects a single prompt string with Human/Assistant format.
        """
        if isinstance(messages, str):
            return f"\n\nHuman: {messages}\n\nAssistant:"

        # Convert message list to Claude text completion format
        formatted_messages = super()._format_messages(messages)
        prompt_parts = []

        for message in formatted_messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                # System messages go at the beginning
                prompt_parts.insert(0, content)
            elif role == "user":
                prompt_parts.append(f"\n\nHuman: {content}")
            elif role == "assistant":
                prompt_parts.append(f"\n\nAssistant: {content}")

        # Ensure prompt ends with Assistant prompt
        prompt = "".join(prompt_parts)
        if not prompt.endswith("\n\nAssistant:"):
            prompt += "\n\nAssistant:"

        return prompt

    def _format_messages_for_messages_api(
        self, messages: str | list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], str | None]:
        """Format messages for Claude Messages API.

        Similar to Anthropic's format but adapted for Bedrock.
        """
        # Use base class formatting first
        base_formatted = super()._format_messages(messages)

        formatted_messages = []
        system_message = None

        for message in base_formatted:
            role = message.get("role")
            content = message.get("content", "")

            if role == "system":
                # Extract system message - Bedrock handles it separately
                if system_message:
                    system_message += f"\n\n{content}"
                else:
                    system_message = content
            else:
                # Add user/assistant messages
                role_str = role if role is not None else "user"
                content_str = content if content is not None else ""
                formatted_messages.append({"role": role_str, "content": content_str})

        # Ensure first message is from user (Claude requirement)
        if not formatted_messages:
            formatted_messages.append({"role": "user", "content": "Hello"})
        elif formatted_messages[0]["role"] != "user":
            formatted_messages.insert(0, {"role": "user", "content": "Hello"})

        return formatted_messages, system_message

    def _convert_tools_for_interference(self, tools: list[dict]) -> list[dict]:
        """Convert CrewAI tool format to Bedrock tool format."""
        from crewai.llms.providers.utils.common import safe_tool_conversion

        bedrock_tools = []

        for tool in tools:
            name, description, parameters = safe_tool_conversion(tool, "Bedrock")

            bedrock_tool = {
                "name": name,
                "description": description,
            }

            if parameters and isinstance(parameters, dict):
                bedrock_tool["input_schema"] = parameters

            bedrock_tools.append(bedrock_tool)

        return bedrock_tools

    def _get_model_or_inference_profile(self, model: str) -> str:
        """Get the appropriate model ID or inference profile for Bedrock invocation.

        Newer Claude models require inference profiles instead of direct model IDs
        when using on-demand throughput.
        """
        # Mapping of newer Claude models to their inference profiles
        inference_profile_mapping = {
            # Claude 3.7 Sonnet (2025) - requires inference profile
            "anthropic.claude-3-7-sonnet-20250219-v1:0": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            # Add other newer models that require inference profiles here
            # Format: "model_id": "inference_profile_id"
        }

        # Check if this model requires an inference profile
        if model in inference_profile_mapping:
            inference_profile = inference_profile_mapping[model]
            logging.info(
                f"Using inference profile {inference_profile} for model {model}"
            )
            return inference_profile

        # For older models or models that don't require inference profiles, use the model ID directly
        return model

    def supports_function_calling(self) -> bool:
        """Check if the model supports function calling."""
        return self.supports_tools

    def supports_stop_words(self) -> bool:
        """Check if the model supports stop words."""
        return True  # Most Bedrock models support stop sequences

    def get_context_window_size(self) -> int:
        """Get the context window size for the model."""
        from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO

        # Context window sizes for common Bedrock models
        context_windows = {
            # Anthropic Claude models
            "anthropic.claude-3-5-sonnet": 200000,
            "anthropic.claude-3-5-haiku": 200000,
            "anthropic.claude-3-opus": 200000,
            "anthropic.claude-3-sonnet": 200000,
            "anthropic.claude-3-haiku": 200000,
            "anthropic.claude-v2": 100000,
            "anthropic.claude-v2:1": 200000,
            "anthropic.claude-instant": 100000,
            # Amazon Titan models
            "amazon.titan-text-express": 8000,
            "amazon.titan-text-lite": 4000,
            # AI21 Jurassic models
            "ai21.j2-ultra": 8192,
            "ai21.j2-mid": 8192,
            # Cohere Command models
            "cohere.command-text": 4096,
            "cohere.command-light-text": 4096,
            # Meta Llama models
            "meta.llama2-13b-chat": 4096,
            "meta.llama2-70b-chat": 4096,
        }

        # Find the best match for the model name
        for model_prefix, size in context_windows.items():
            if self.model.startswith(model_prefix):
                return int(size * CONTEXT_WINDOW_USAGE_RATIO)

        # Default context window size
        return int(8192 * CONTEXT_WINDOW_USAGE_RATIO)

    def _extract_bedrock_token_usage(
        self, response_body: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract token usage from Bedrock response."""
        usage = response_body.get("usage", {})

        # Handle different response formats
        if "input_tokens" in usage and "output_tokens" in usage:
            # Messages API format
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            return {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        if (
            "prompt_token_count" in response_body
            and "generation_token_count" in response_body
        ):
            # Text completion format
            prompt_tokens = response_body.get("prompt_token_count", 0)
            completion_tokens = response_body.get("generation_token_count", 0)
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

        return {"total_tokens": 0}
