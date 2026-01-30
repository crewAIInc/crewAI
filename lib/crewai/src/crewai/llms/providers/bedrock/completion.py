from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
import json
import logging
import os
from typing import TYPE_CHECKING, Any, TypedDict, cast

from pydantic import BaseModel
from typing_extensions import Required

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.pydantic_schema_utils import generate_model_description
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import (
        GuardrailConfigurationTypeDef,
        GuardrailStreamConfigurationTypeDef,
        InferenceConfigurationTypeDef,
        MessageOutputTypeDef,
        MessageTypeDef,
        SystemContentBlockTypeDef,
        TokenUsageTypeDef,
        ToolConfigurationTypeDef,
        ToolTypeDef,
    )

    from crewai.llms.hooks.base import BaseInterceptor


try:
    from boto3.session import Session
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    raise ImportError(
        'AWS Bedrock native provider not available, to install: uv add "crewai[bedrock]"'
    ) from None


STRUCTURED_OUTPUT_TOOL_NAME = "structured_output"


def _preprocess_structured_data(
    data: dict[str, Any], response_model: type[BaseModel]
) -> dict[str, Any]:
    """Preprocess structured data to handle common LLM output format issues.

    Some models (especially Claude on Bedrock) may return array fields as
    markdown-formatted strings instead of proper JSON arrays. This function
    attempts to convert such strings to arrays before validation.

    Args:
        data: The raw structured data from the tool response
        response_model: The Pydantic model class to validate against

    Returns:
        Preprocessed data with string-to-array conversions where needed
    """
    import re
    from typing import get_origin

    # Get model field annotations
    model_fields = response_model.model_fields

    processed_data = dict(data)

    for field_name, field_info in model_fields.items():
        if field_name not in processed_data:
            continue

        value = processed_data[field_name]

        # Check if the field expects a list type
        annotation = field_info.annotation
        origin = get_origin(annotation)

        # Handle list[X] or List[X] types
        is_list_type = origin is list or (
            origin is not None and str(origin).startswith("list")
        )

        if is_list_type and isinstance(value, str):
            # Try to parse markdown-style bullet points or numbered lists
            lines = value.strip().split("\n")
            parsed_items = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Remove common bullet point prefixes
                # Matches: "- item", "* item", "• item", "1. item", "1) item"
                cleaned = re.sub(r"^[-*•]\s*", "", line)
                cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
                cleaned = cleaned.strip()

                if cleaned:
                    parsed_items.append(cleaned)

            if parsed_items:
                processed_data[field_name] = parsed_items
                logging.debug(
                    f"Converted markdown-formatted string to list for field '{field_name}': "
                    f"{len(parsed_items)} items"
                )

    return processed_data


try:
    from aiobotocore.session import (  # type: ignore[import-untyped]
        get_session as get_aiobotocore_session,
    )

    AIOBOTOCORE_AVAILABLE = True
except ImportError:
    AIOBOTOCORE_AVAILABLE = False
    get_aiobotocore_session = None


if TYPE_CHECKING:

    class EnhancedInferenceConfigurationTypeDef(
        InferenceConfigurationTypeDef, total=False
    ):
        """Extended InferenceConfigurationTypeDef with topK support.

        AWS Bedrock supports topK for Claude models, but it's not in the boto3 type stubs.
        This extends the base type to include topK while maintaining all other fields.
        """

        topK: int  # noqa: N815 - AWS API uses topK naming

else:

    class EnhancedInferenceConfigurationTypeDef(TypedDict, total=False):
        """Extended InferenceConfigurationTypeDef with topK support.

        AWS Bedrock supports topK for Claude models, but it's not in the boto3 type stubs.
        This extends the base type to include topK while maintaining all other fields.
        """

        maxTokens: int
        temperature: float
        topP: float
        stopSequences: list[str]
        topK: int


class ToolInputSchema(TypedDict):
    """Type definition for tool input schema in Converse API."""

    json: dict[str, Any]


class ToolSpec(TypedDict, total=False):
    """Type definition for tool specification in Converse API."""

    name: Required[str]
    description: Required[str]
    inputSchema: ToolInputSchema


class ConverseToolTypeDef(TypedDict):
    """Type definition for a Converse API tool."""

    toolSpec: ToolSpec


class BedrockConverseRequestBody(TypedDict, total=False):
    """Type definition for AWS Bedrock Converse API request body.

    Based on AWS Bedrock Converse API specification.
    """

    inferenceConfig: Required[EnhancedInferenceConfigurationTypeDef]
    system: list[SystemContentBlockTypeDef]
    toolConfig: ToolConfigurationTypeDef
    guardrailConfig: GuardrailConfigurationTypeDef
    additionalModelRequestFields: dict[str, Any]
    additionalModelResponseFieldPaths: list[str]


class BedrockConverseStreamRequestBody(TypedDict, total=False):
    """Type definition for AWS Bedrock Converse Stream API request body.

    Based on AWS Bedrock Converse Stream API specification.
    """

    inferenceConfig: Required[EnhancedInferenceConfigurationTypeDef]
    system: list[SystemContentBlockTypeDef]
    toolConfig: ToolConfigurationTypeDef
    guardrailConfig: GuardrailStreamConfigurationTypeDef
    additionalModelRequestFields: dict[str, Any]
    additionalModelResponseFieldPaths: list[str]


class BedrockCompletion(BaseLLM):
    """AWS Bedrock native completion implementation using the Converse API.

    This class provides direct integration with AWS Bedrock using the modern
    Converse API, which provides a unified interface across all Bedrock models.

    Features:
    - Full tool calling support with proper conversation continuation
    - Streaming and non-streaming responses with comprehensive event handling
    - Guardrail configuration for content filtering
    - Model-specific parameters via additionalModelRequestFields
    - Custom response field extraction
    - Proper error handling for all AWS exception types
    - Token usage tracking and stop reason logging
    - Support for both text and tool use content blocks

    The implementation follows AWS Bedrock Converse API best practices including:
    - Proper tool use ID tracking for multi-turn tool conversations
    - Complete streaming event handling (messageStart, contentBlockStart, etc.)
    - Response metadata and trace information capture
    - Model-specific conversation format handling (e.g., Cohere requirements)
    """

    def __init__(
        self,
        model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        region_name: str = "us-east-1",
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: Sequence[str] | None = None,
        stream: bool = False,
        guardrail_config: dict[str, Any] | None = None,
        additional_model_request_fields: dict[str, Any] | None = None,
        additional_model_response_field_paths: list[str] | None = None,
        interceptor: BaseInterceptor[Any, Any] | None = None,
        response_format: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize AWS Bedrock completion client.

        Args:
            model: The Bedrock model ID to use
            aws_access_key_id: AWS access key (defaults to environment variable)
            aws_secret_access_key: AWS secret key (defaults to environment variable)
            aws_session_token: AWS session token for temporary credentials
            region_name: AWS region name
            temperature: Sampling temperature for response generation
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter (Claude models only)
            stop_sequences: List of sequences that stop generation
            stream: Whether to use streaming responses
            guardrail_config: Guardrail configuration for content filtering
            additional_model_request_fields: Model-specific request parameters
            additional_model_response_field_paths: Custom response field paths
            interceptor: HTTP interceptor (not yet supported for Bedrock).
            response_format: Pydantic model for structured output. Used as default when
                           response_model is not passed to call()/acall() methods.
            **kwargs: Additional parameters
        """
        if interceptor is not None:
            raise NotImplementedError(
                "HTTP interceptors are not yet supported for AWS Bedrock provider. "
                "Interceptors are currently supported for OpenAI and Anthropic providers only."
            )

        # Extract provider from kwargs to avoid duplicate argument
        kwargs.pop("provider", None)

        super().__init__(
            model=model,
            temperature=temperature,
            stop=stop_sequences or [],
            provider="bedrock",
            **kwargs,
        )

        # Initialize Bedrock client with proper configuration
        session = Session(
            aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret_access_key
            or os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=aws_session_token or os.getenv("AWS_SESSION_TOKEN"),
            region_name=region_name,
        )

        # Configure client with timeouts and retries following AWS best practices
        config = Config(
            read_timeout=300,
            retries={
                "max_attempts": 3,
                "mode": "adaptive",
            },
            tcp_keepalive=True,
        )

        self.client = session.client("bedrock-runtime", config=config)
        self.region_name = region_name

        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
        self.aws_session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")

        self._async_exit_stack = AsyncExitStack() if AIOBOTOCORE_AVAILABLE else None
        self._async_client_initialized = False

        # Store completion parameters
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.stream = stream
        self.stop_sequences = stop_sequences
        self.response_format = response_format

        # Store advanced features (optional)
        self.guardrail_config = guardrail_config
        self.additional_model_request_fields = additional_model_request_fields
        self.additional_model_response_field_paths = (
            additional_model_response_field_paths
        )

        # Model-specific settings
        self.is_claude_model = "claude" in model.lower()
        self.supports_tools = True  # Converse API supports tools for most models
        self.supports_streaming = True

        # Handle inference profiles for newer models
        self.model_id = model

    @property
    def stop(self) -> list[str]:
        """Get stop sequences sent to the API."""
        return [] if self.stop_sequences is None else list(self.stop_sequences)

    @stop.setter
    def stop(self, value: Sequence[str] | str | None) -> None:
        """Set stop sequences.

        Synchronizes stop_sequences to ensure values set by CrewAgentExecutor
        are properly sent to the Bedrock API.

        Args:
            value: Stop sequences as a Sequence, single string, or None
        """
        if value is None:
            self.stop_sequences = []
        elif isinstance(value, str):
            self.stop_sequences = [value]
        elif isinstance(value, Sequence):
            self.stop_sequences = list(value)
        else:
            self.stop_sequences = []

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[Any, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Call AWS Bedrock Converse API."""
        effective_response_model = response_model or self.response_format

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

            # Format messages for Converse API
            formatted_messages, system_message = self._format_messages_for_converse(
                messages
            )

            if not self._invoke_before_llm_call_hooks(formatted_messages, from_agent):
                raise ValueError("LLM call blocked by before_llm_call hook")

            # Prepare request body
            body: BedrockConverseRequestBody = {
                "inferenceConfig": self._get_inference_config(),
            }

            # Add system message if present
            if system_message:
                body["system"] = cast(
                    "list[SystemContentBlockTypeDef]",
                    cast(object, [{"text": system_message}]),
                )

            # Add tool config if present or if messages contain tool content
            # Bedrock requires toolConfig when messages have toolUse/toolResult
            if tools:
                tool_config: ToolConfigurationTypeDef = {
                    "tools": cast(
                        "Sequence[ToolTypeDef]",
                        cast(object, self._format_tools_for_converse(tools)),
                    )
                }
                body["toolConfig"] = tool_config
            elif self._messages_contain_tool_content(formatted_messages):
                # Create minimal toolConfig from tool history in messages
                tools_from_history = self._extract_tools_from_message_history(
                    formatted_messages
                )
                if tools_from_history:
                    body["toolConfig"] = cast(
                        "ToolConfigurationTypeDef",
                        cast(object, {"tools": tools_from_history}),
                    )

            # Add optional advanced features if configured
            if self.guardrail_config:
                guardrail_config: GuardrailConfigurationTypeDef = cast(
                    "GuardrailConfigurationTypeDef", cast(object, self.guardrail_config)
                )
                body["guardrailConfig"] = guardrail_config

            if self.additional_model_request_fields:
                body["additionalModelRequestFields"] = (
                    self.additional_model_request_fields
                )

            if self.additional_model_response_field_paths:
                body["additionalModelResponseFieldPaths"] = (
                    self.additional_model_response_field_paths
                )

            if self.stream:
                return self._handle_streaming_converse(
                    formatted_messages,
                    body,
                    available_functions,
                    from_task,
                    from_agent,
                    effective_response_model,
                )

            return self._handle_converse(
                formatted_messages,
                body,
                available_functions,
                from_task,
                from_agent,
                effective_response_model,
            )

        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"AWS Bedrock API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    async def acall(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[Any, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Async call to AWS Bedrock Converse API.

        Args:
            messages: Input messages as string or list of message dicts.
            tools: Optional list of tool definitions.
            callbacks: Optional list of callback handlers.
            available_functions: Optional dict mapping function names to callables.
            from_task: Optional task context for events.
            from_agent: Optional agent context for events.
            response_model: Optional Pydantic model for structured output.

        Returns:
            Generated text response or structured output.

        Raises:
            NotImplementedError: If aiobotocore is not installed.
            LLMContextLengthExceededError: If context window is exceeded.
        """
        effective_response_model = response_model or self.response_format

        if not AIOBOTOCORE_AVAILABLE:
            raise NotImplementedError(
                "Async support for AWS Bedrock requires aiobotocore. "
                'Install with: uv add "crewai[bedrock-async]"'
            )

        try:
            self._emit_call_started_event(
                messages=messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
            )

            formatted_messages, system_message = self._format_messages_for_converse(
                messages
            )

            body: BedrockConverseRequestBody = {
                "inferenceConfig": self._get_inference_config(),
            }

            if system_message:
                body["system"] = cast(
                    "list[SystemContentBlockTypeDef]",
                    cast(object, [{"text": system_message}]),
                )

            # Add tool config if present or if messages contain tool content
            # Bedrock requires toolConfig when messages have toolUse/toolResult
            if tools:
                tool_config: ToolConfigurationTypeDef = {
                    "tools": cast(
                        "Sequence[ToolTypeDef]",
                        cast(object, self._format_tools_for_converse(tools)),
                    )
                }
                body["toolConfig"] = tool_config
            elif self._messages_contain_tool_content(formatted_messages):
                # Create minimal toolConfig from tool history in messages
                tools_from_history = self._extract_tools_from_message_history(
                    formatted_messages
                )
                if tools_from_history:
                    body["toolConfig"] = cast(
                        "ToolConfigurationTypeDef",
                        cast(object, {"tools": tools_from_history}),
                    )

            if self.guardrail_config:
                guardrail_config: GuardrailConfigurationTypeDef = cast(
                    "GuardrailConfigurationTypeDef", cast(object, self.guardrail_config)
                )
                body["guardrailConfig"] = guardrail_config

            if self.additional_model_request_fields:
                body["additionalModelRequestFields"] = (
                    self.additional_model_request_fields
                )

            if self.additional_model_response_field_paths:
                body["additionalModelResponseFieldPaths"] = (
                    self.additional_model_response_field_paths
                )

            if self.stream:
                return await self._ahandle_streaming_converse(
                    formatted_messages,
                    body,
                    available_functions,
                    from_task,
                    from_agent,
                    effective_response_model,
                )

            return await self._ahandle_converse(
                formatted_messages,
                body,
                available_functions,
                from_task,
                from_agent,
                effective_response_model,
            )

        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"AWS Bedrock API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    def _handle_converse(
        self,
        messages: list[LLMMessage],
        body: BedrockConverseRequestBody,
        available_functions: Mapping[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle non-streaming converse API call following AWS best practices."""
        if response_model:
            # Check if structured_output tool already exists (from a previous recursive call)
            existing_tool_config = body.get("toolConfig")
            existing_tools: list[Any] = []
            structured_output_already_exists = False

            if existing_tool_config:
                existing_tools = list(existing_tool_config.get("tools", []))
                for tool in existing_tools:
                    tool_spec = tool.get("toolSpec", {})
                    if tool_spec.get("name") == STRUCTURED_OUTPUT_TOOL_NAME:
                        structured_output_already_exists = True
                        break

            if not structured_output_already_exists:
                structured_tool: ConverseToolTypeDef = {
                    "toolSpec": {
                        "name": STRUCTURED_OUTPUT_TOOL_NAME,
                        "description": (
                            "Use this tool to provide your final structured response. "
                            "Call this tool when you have gathered all necessary information "
                            "and are ready to provide the final answer in the required format."
                        ),
                        "inputSchema": {
                            "json": generate_model_description(response_model)
                            .get("json_schema", {})
                            .get("schema", {})
                        },
                    }
                }

                if existing_tools:
                    existing_tools.append(structured_tool)
                    body["toolConfig"] = cast(
                        "ToolConfigurationTypeDef",
                        cast(object, {"tools": existing_tools}),
                    )
                else:
                    # No existing tools, use only structured_output with forced toolChoice
                    body["toolConfig"] = cast(
                        "ToolConfigurationTypeDef",
                        cast(
                            object,
                            {
                                "tools": [structured_tool],
                                "toolChoice": {
                                    "tool": {"name": STRUCTURED_OUTPUT_TOOL_NAME}
                                },
                            },
                        ),
                    )

        try:
            if not messages:
                raise ValueError("Messages cannot be empty")

            # Ensure we have valid message structure
            for i, msg in enumerate(messages):
                if (
                    not isinstance(msg, dict)
                    or "role" not in msg
                    or "content" not in msg
                ):
                    raise ValueError(f"Invalid message format at index {i}")

            # Call Bedrock Converse API with proper error handling
            response = self.client.converse(
                modelId=self.model_id,
                messages=cast(
                    "Sequence[MessageTypeDef | MessageOutputTypeDef]",
                    cast(object, messages),
                ),
                **body,
            )

            # Track token usage according to AWS response format
            if "usage" in response:
                self._track_token_usage_internal(response["usage"])

            stop_reason = response.get("stopReason")
            if stop_reason:
                logging.debug(f"Response stop reason: {stop_reason}")
                if stop_reason == "max_tokens":
                    logging.warning("Response truncated due to max_tokens limit")
                elif stop_reason == "content_filtered":
                    logging.warning("Response was filtered due to content policy")

            # Extract content following AWS response structure
            output = response.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])

            if not content:
                logging.warning("No content in Bedrock response")
                return (
                    "I apologize, but I received an empty response. Please try again."
                )

            # If there are tool uses but no available_functions, return them for the executor to handle
            tool_uses = [block["toolUse"] for block in content if "toolUse" in block]

            # Check for structured_output tool call first
            if response_model and tool_uses:
                for tool_use in tool_uses:
                    if tool_use.get("name") == STRUCTURED_OUTPUT_TOOL_NAME:
                        structured_data = tool_use.get("input", {})
                        structured_data = _preprocess_structured_data(
                            structured_data, response_model
                        )
                        try:
                            result = response_model.model_validate(structured_data)
                            self._emit_call_completed_event(
                                response=result.model_dump_json(),
                                call_type=LLMCallType.LLM_CALL,
                                from_task=from_task,
                                from_agent=from_agent,
                                messages=messages,
                            )
                            return result
                        except Exception as e:
                            error_msg = (
                                f"Failed to validate {STRUCTURED_OUTPUT_TOOL_NAME} tool response "
                                f"with model {response_model.__name__}: {e}"
                            )
                            logging.error(error_msg)
                            raise ValueError(error_msg) from e

            # Filter out structured_output from tool_uses returned to executor
            non_structured_output_tool_uses = [
                tu for tu in tool_uses if tu.get("name") != STRUCTURED_OUTPUT_TOOL_NAME
            ]

            if non_structured_output_tool_uses and not available_functions:
                self._emit_call_completed_event(
                    response=non_structured_output_tool_uses,
                    call_type=LLMCallType.TOOL_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=messages,
                )
                return non_structured_output_tool_uses

            # Process content blocks and handle tool use correctly
            text_content = ""

            for content_block in content:
                # Handle text content
                if "text" in content_block:
                    text_content += content_block["text"]

                # Handle tool use - corrected structure according to AWS API docs
                elif "toolUse" in content_block and available_functions:
                    tool_use_block = content_block["toolUse"]
                    tool_use_id = tool_use_block.get("toolUseId")
                    function_name = tool_use_block["name"]
                    function_args = tool_use_block.get("input", {})

                    if function_name == STRUCTURED_OUTPUT_TOOL_NAME:
                        continue

                    logging.debug(
                        f"Tool use requested: {function_name} with ID {tool_use_id}"
                    )

                    # Execute the tool
                    tool_result = self._handle_tool_execution(
                        function_name=function_name,
                        function_args=function_args,
                        available_functions=dict(available_functions),
                        from_task=from_task,
                        from_agent=from_agent,
                    )

                    if tool_result is not None:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [{"toolUse": tool_use_block}],
                            }
                        )

                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "toolResult": {
                                            "toolUseId": tool_use_id,
                                            "content": [{"text": str(tool_result)}],
                                        }
                                    }
                                ],
                            }
                        )

                        return self._handle_converse(
                            messages,
                            body,
                            available_functions,
                            from_task,
                            from_agent,
                            response_model,
                        )

            # Apply stop sequences if configured
            text_content = self._apply_stop_words(text_content)

            # Validate final response
            if not text_content or text_content.strip() == "":
                logging.warning("Extracted empty text content from Bedrock response")
                text_content = "I apologize, but I couldn't generate a proper response. Please try again."

            self._emit_call_completed_event(
                response=text_content,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=messages,
            )

            return self._invoke_after_llm_call_hooks(
                messages,
                text_content,
                from_agent,
            )

        except ClientError as e:
            # Handle all AWS ClientError exceptions as per documentation
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))

            # Log the specific error for debugging
            logging.error(f"AWS Bedrock ClientError ({error_code}): {error_msg}")

            # Handle specific error codes as documented
            if error_code == "ValidationException":
                # This is the error we're seeing with Cohere
                if "last turn" in error_msg and "user message" in error_msg:
                    raise ValueError(
                        f"Conversation format error: {error_msg}. Check message alternation."
                    ) from e
                raise ValueError(f"Request validation failed: {error_msg}") from e
            if error_code == "AccessDeniedException":
                raise PermissionError(
                    f"Access denied to model {self.model_id}: {error_msg}"
                ) from e
            if error_code == "ResourceNotFoundException":
                raise ValueError(f"Model {self.model_id} not found: {error_msg}") from e
            if error_code == "ThrottlingException":
                raise RuntimeError(
                    f"API throttled, please retry later: {error_msg}"
                ) from e
            if error_code == "ModelTimeoutException":
                raise TimeoutError(f"Model request timed out: {error_msg}") from e
            if error_code == "ServiceQuotaExceededException":
                raise RuntimeError(f"Service quota exceeded: {error_msg}") from e
            if error_code == "ModelNotReadyException":
                raise RuntimeError(
                    f"Model {self.model_id} not ready: {error_msg}"
                ) from e
            if error_code == "ModelErrorException":
                raise RuntimeError(f"Model error: {error_msg}") from e
            if error_code == "InternalServerException":
                raise RuntimeError(f"Internal server error: {error_msg}") from e
            if error_code == "ServiceUnavailableException":
                raise RuntimeError(f"Service unavailable: {error_msg}") from e

            raise RuntimeError(f"Bedrock API error ({error_code}): {error_msg}") from e

        except BotoCoreError as e:
            error_msg = f"Bedrock connection error: {e}"
            logging.error(error_msg)
            raise ConnectionError(error_msg) from e
        except Exception as e:
            # Catch any other unexpected errors
            error_msg = f"Unexpected error in Bedrock converse call: {e}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _handle_streaming_converse(
        self,
        messages: list[LLMMessage],
        body: BedrockConverseRequestBody,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        """Handle streaming converse API call with comprehensive event handling."""
        if response_model:
            # Check if structured_output tool already exists (from a previous recursive call)
            existing_tool_config = body.get("toolConfig")
            existing_tools: list[Any] = []
            structured_output_already_exists = False

            if existing_tool_config:
                existing_tools = list(existing_tool_config.get("tools", []))
                # Check if structured_output tool is already in the tools list
                for tool in existing_tools:
                    tool_spec = tool.get("toolSpec", {})
                    if tool_spec.get("name") == STRUCTURED_OUTPUT_TOOL_NAME:
                        structured_output_already_exists = True
                        break

            if not structured_output_already_exists:
                structured_tool: ConverseToolTypeDef = {
                    "toolSpec": {
                        "name": STRUCTURED_OUTPUT_TOOL_NAME,
                        "description": (
                            "Use this tool to provide your final structured response. "
                            "Call this tool when you have gathered all necessary information "
                            "and are ready to provide the final answer in the required format."
                        ),
                        "inputSchema": {
                            "json": generate_model_description(response_model)
                            .get("json_schema", {})
                            .get("schema", {})
                        },
                    }
                }

                if existing_tools:
                    # Append structured_output to existing tools, don't force toolChoice
                    existing_tools.append(structured_tool)
                    body["toolConfig"] = cast(
                        "ToolConfigurationTypeDef",
                        cast(object, {"tools": existing_tools}),
                    )
                else:
                    # No existing tools, use only structured_output with forced toolChoice
                    body["toolConfig"] = cast(
                        "ToolConfigurationTypeDef",
                        cast(
                            object,
                            {
                                "tools": [structured_tool],
                                "toolChoice": {
                                    "tool": {"name": STRUCTURED_OUTPUT_TOOL_NAME}
                                },
                            },
                        ),
                    )

        full_response = ""
        current_tool_use: dict[str, Any] | None = None
        tool_use_id: str | None = None
        tool_use_index = 0
        accumulated_tool_input = ""

        try:
            response = self.client.converse_stream(
                modelId=self.model_id,
                messages=cast(
                    "Sequence[MessageTypeDef | MessageOutputTypeDef]",
                    cast(object, messages),
                ),
                **body,  # type: ignore[arg-type]
            )

            stream = response.get("stream")
            response_id = None
            if stream:
                for event in stream:
                    if "messageStart" in event:
                        role = event["messageStart"].get("role")
                        logging.debug(f"Streaming message started with role: {role}")

                    elif "contentBlockStart" in event:
                        start = event["contentBlockStart"].get("start", {})
                        content_block_index = event["contentBlockStart"].get(
                            "contentBlockIndex", 0
                        )
                        if "toolUse" in start:
                            tool_use_block = start["toolUse"]
                            current_tool_use = cast(dict[str, Any], tool_use_block)
                            tool_use_id = current_tool_use.get("toolUseId")
                            tool_use_index = content_block_index
                            accumulated_tool_input = ""
                            self._emit_stream_chunk_event(
                                chunk="",
                                from_task=from_task,
                                from_agent=from_agent,
                                tool_call={
                                    "id": tool_use_id or "",
                                    "function": {
                                        "name": current_tool_use.get("name", ""),
                                        "arguments": "",
                                    },
                                    "type": "function",
                                    "index": tool_use_index,
                                },
                                call_type=LLMCallType.TOOL_CALL,
                                response_id=response_id,
                            )
                        logging.debug(
                            f"Tool use started in stream: {json.dumps(current_tool_use)} (ID: {tool_use_id})"
                        )

                    elif "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"]["delta"]
                        if "text" in delta:
                            text_chunk = delta["text"]
                            logging.debug(f"Streaming text chunk: {text_chunk[:50]}...")
                            full_response += text_chunk
                            self._emit_stream_chunk_event(
                                chunk=text_chunk,
                                from_task=from_task,
                                from_agent=from_agent,
                                response_id=response_id,
                            )
                        elif "toolUse" in delta and current_tool_use:
                            tool_input = delta["toolUse"].get("input", "")
                            if tool_input:
                                accumulated_tool_input += tool_input
                                logging.debug(f"Tool input delta: {tool_input}")
                                self._emit_stream_chunk_event(
                                    chunk=tool_input,
                                    from_task=from_task,
                                    from_agent=from_agent,
                                    tool_call={
                                        "id": tool_use_id or "",
                                        "function": {
                                            "name": current_tool_use.get("name", ""),
                                            "arguments": accumulated_tool_input,
                                        },
                                        "type": "function",
                                        "index": tool_use_index,
                                    },
                                    call_type=LLMCallType.TOOL_CALL,
                                    response_id=response_id,
                                )
                    elif "contentBlockStop" in event:
                        logging.debug("Content block stopped in stream")
                        if current_tool_use:
                            function_name = current_tool_use["name"]
                            function_args = cast(
                                dict[str, Any], current_tool_use.get("input", {})
                            )

                            # Check if this is the structured_output tool
                            if (
                                function_name == STRUCTURED_OUTPUT_TOOL_NAME
                                and response_model
                            ):
                                function_args = _preprocess_structured_data(
                                    function_args, response_model
                                )
                                try:
                                    result = response_model.model_validate(
                                        function_args
                                    )
                                    self._emit_call_completed_event(
                                        response=result.model_dump_json(),
                                        call_type=LLMCallType.LLM_CALL,
                                        from_task=from_task,
                                        from_agent=from_agent,
                                        messages=messages,
                                    )
                                    return result  # type: ignore[return-value]
                                except Exception as e:
                                    error_msg = (
                                        f"Failed to validate {STRUCTURED_OUTPUT_TOOL_NAME} tool response "
                                        f"with model {response_model.__name__}: {e}"
                                    )
                                    logging.error(error_msg)
                                    raise ValueError(error_msg) from e

                            # Handle regular tool execution
                            if available_functions:
                                tool_result = self._handle_tool_execution(
                                    function_name=function_name,
                                    function_args=function_args,
                                    available_functions=available_functions,
                                    from_task=from_task,
                                    from_agent=from_agent,
                                )
                                if tool_result is not None and tool_use_id:
                                    messages.append(
                                        {
                                            "role": "assistant",
                                            "content": [{"toolUse": current_tool_use}],
                                        }
                                    )
                                    messages.append(
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "toolResult": {
                                                        "toolUseId": tool_use_id,
                                                        "content": [
                                                            {"text": str(tool_result)}
                                                        ],
                                                    }
                                                }
                                            ],
                                        }
                                    )
                                    return self._handle_converse(
                                        messages,
                                        body,
                                        available_functions,
                                        from_task,
                                        from_agent,
                                        response_model,
                                    )
                            current_tool_use = None
                            tool_use_id = None
                    elif "messageStop" in event:
                        stop_reason = event["messageStop"].get("stopReason")
                        logging.debug(f"Streaming message stopped: {stop_reason}")
                        if stop_reason == "max_tokens":
                            logging.warning(
                                "Streaming response truncated due to max_tokens"
                            )
                        elif stop_reason == "content_filtered":
                            logging.warning(
                                "Streaming response filtered due to content policy"
                            )
                            break
                    elif "metadata" in event:
                        metadata = event["metadata"]
                        if "usage" in metadata:
                            usage_metrics = metadata["usage"]
                            self._track_token_usage_internal(usage_metrics)
                            logging.debug(f"Token usage: {usage_metrics}")
                            if "trace" in metadata:
                                logging.debug(
                                    f"Trace information available: {metadata['trace']}"
                                )

        except ClientError as e:
            error_msg = self._handle_client_error(e)
            raise RuntimeError(error_msg) from e
        except BotoCoreError as e:
            error_msg = f"Bedrock streaming connection error: {e}"
            logging.error(error_msg)
            raise ConnectionError(error_msg) from e

        full_response = self._apply_stop_words(full_response)

        if not full_response or full_response.strip() == "":
            logging.warning("Bedrock streaming returned empty content, using fallback")
            full_response = (
                "I apologize, but I couldn't generate a response. Please try again."
            )

        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=messages,
        )

        return full_response

    async def _ensure_async_client(self) -> Any:
        """Ensure async client is initialized and return it."""
        if not self._async_client_initialized and get_aiobotocore_session:
            if self._async_exit_stack is None:
                raise RuntimeError(
                    "Async exit stack not initialized - aiobotocore not available"
                )
            session = get_aiobotocore_session()
            client = await self._async_exit_stack.enter_async_context(
                session.create_client(
                    "bedrock-runtime",
                    region_name=self.region_name,
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    aws_session_token=self.aws_session_token,
                )
            )
            self._async_client = client
            self._async_client_initialized = True
        return self._async_client

    async def _ahandle_converse(
        self,
        messages: list[LLMMessage],
        body: BedrockConverseRequestBody,
        available_functions: Mapping[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle async non-streaming converse API call."""
        if response_model:
            # Check if structured_output tool already exists (from a previous recursive call)
            existing_tool_config = body.get("toolConfig")
            existing_tools: list[Any] = []
            structured_output_already_exists = False

            if existing_tool_config:
                existing_tools = list(existing_tool_config.get("tools", []))
                # Check if structured_output tool is already in the tools list
                for tool in existing_tools:
                    tool_spec = tool.get("toolSpec", {})
                    if tool_spec.get("name") == STRUCTURED_OUTPUT_TOOL_NAME:
                        structured_output_already_exists = True
                        break

            if not structured_output_already_exists:
                structured_tool: ConverseToolTypeDef = {
                    "toolSpec": {
                        "name": STRUCTURED_OUTPUT_TOOL_NAME,
                        "description": (
                            "Use this tool to provide your final structured response. "
                            "Call this tool when you have gathered all necessary information "
                            "and are ready to provide the final answer in the required format."
                        ),
                        "inputSchema": {
                            "json": generate_model_description(response_model)
                            .get("json_schema", {})
                            .get("schema", {})
                        },
                    }
                }

                if existing_tools:
                    # Append structured_output to existing tools, don't force toolChoice
                    existing_tools.append(structured_tool)
                    body["toolConfig"] = cast(
                        "ToolConfigurationTypeDef",
                        cast(object, {"tools": existing_tools}),
                    )
                else:
                    # No existing tools, use only structured_output with forced toolChoice
                    body["toolConfig"] = cast(
                        "ToolConfigurationTypeDef",
                        cast(
                            object,
                            {
                                "tools": [structured_tool],
                                "toolChoice": {
                                    "tool": {"name": STRUCTURED_OUTPUT_TOOL_NAME}
                                },
                            },
                        ),
                    )

        try:
            if not messages:
                raise ValueError("Messages cannot be empty")

            for i, msg in enumerate(messages):
                if (
                    not isinstance(msg, dict)
                    or "role" not in msg
                    or "content" not in msg
                ):
                    raise ValueError(f"Invalid message format at index {i}")

            async_client = await self._ensure_async_client()
            response = await async_client.converse(
                modelId=self.model_id,
                messages=cast(
                    "Sequence[MessageTypeDef | MessageOutputTypeDef]",
                    cast(object, messages),
                ),
                **body,
            )

            if "usage" in response:
                self._track_token_usage_internal(response["usage"])

            stop_reason = response.get("stopReason")
            if stop_reason:
                logging.debug(f"Response stop reason: {stop_reason}")
                if stop_reason == "max_tokens":
                    logging.warning("Response truncated due to max_tokens limit")
                elif stop_reason == "content_filtered":
                    logging.warning("Response was filtered due to content policy")

            output = response.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])

            if not content:
                logging.warning("No content in Bedrock response")
                return (
                    "I apologize, but I received an empty response. Please try again."
                )

            # If there are tool uses but no available_functions, return them for the executor to handle
            tool_uses = [block["toolUse"] for block in content if "toolUse" in block]

            # Check for structured_output tool call first
            if response_model and tool_uses:
                for tool_use in tool_uses:
                    if tool_use.get("name") == STRUCTURED_OUTPUT_TOOL_NAME:
                        structured_data = tool_use.get("input", {})
                        structured_data = _preprocess_structured_data(
                            structured_data, response_model
                        )
                        try:
                            result = response_model.model_validate(structured_data)
                            self._emit_call_completed_event(
                                response=result.model_dump_json(),
                                call_type=LLMCallType.LLM_CALL,
                                from_task=from_task,
                                from_agent=from_agent,
                                messages=messages,
                            )
                            return result
                        except Exception as e:
                            error_msg = (
                                f"Failed to validate {STRUCTURED_OUTPUT_TOOL_NAME} tool response "
                                f"with model {response_model.__name__}: {e}"
                            )
                            logging.error(error_msg)
                            raise ValueError(error_msg) from e

            # Filter out structured_output from tool_uses returned to executor
            non_structured_output_tool_uses = [
                tu for tu in tool_uses if tu.get("name") != STRUCTURED_OUTPUT_TOOL_NAME
            ]

            if non_structured_output_tool_uses and not available_functions:
                self._emit_call_completed_event(
                    response=non_structured_output_tool_uses,
                    call_type=LLMCallType.TOOL_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=messages,
                )
                return non_structured_output_tool_uses

            text_content = ""

            for content_block in content:
                if "text" in content_block:
                    text_content += content_block["text"]

                elif "toolUse" in content_block and available_functions:
                    tool_use_block = content_block["toolUse"]
                    tool_use_id = tool_use_block.get("toolUseId")
                    function_name = tool_use_block["name"]
                    function_args = tool_use_block.get("input", {})

                    # Skip structured_output - it's handled above
                    if function_name == STRUCTURED_OUTPUT_TOOL_NAME:
                        continue

                    logging.debug(
                        f"Tool use requested: {function_name} with ID {tool_use_id}"
                    )

                    tool_result = self._handle_tool_execution(
                        function_name=function_name,
                        function_args=function_args,
                        available_functions=dict(available_functions),
                        from_task=from_task,
                        from_agent=from_agent,
                    )

                    if tool_result is not None:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [{"toolUse": tool_use_block}],
                            }
                        )

                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "toolResult": {
                                            "toolUseId": tool_use_id,
                                            "content": [{"text": str(tool_result)}],
                                        }
                                    }
                                ],
                            }
                        )

                        return await self._ahandle_converse(
                            messages,
                            body,
                            available_functions,
                            from_task,
                            from_agent,
                            response_model,
                        )

            text_content = self._apply_stop_words(text_content)

            if not text_content or text_content.strip() == "":
                logging.warning("Extracted empty text content from Bedrock response")
                text_content = "I apologize, but I couldn't generate a proper response. Please try again."

            self._emit_call_completed_event(
                response=text_content,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=messages,
            )

            return text_content

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            logging.error(f"AWS Bedrock ClientError ({error_code}): {error_msg}")

            if error_code == "ValidationException":
                if "last turn" in error_msg and "user message" in error_msg:
                    raise ValueError(
                        f"Conversation format error: {error_msg}. Check message alternation."
                    ) from e
                raise ValueError(f"Request validation failed: {error_msg}") from e
            if error_code == "AccessDeniedException":
                raise PermissionError(
                    f"Access denied to model {self.model_id}: {error_msg}"
                ) from e
            if error_code == "ResourceNotFoundException":
                raise ValueError(f"Model {self.model_id} not found: {error_msg}") from e
            if error_code == "ThrottlingException":
                raise RuntimeError(
                    f"API throttled, please retry later: {error_msg}"
                ) from e
            if error_code == "ModelTimeoutException":
                raise TimeoutError(f"Model request timed out: {error_msg}") from e
            if error_code == "ServiceQuotaExceededException":
                raise RuntimeError(f"Service quota exceeded: {error_msg}") from e
            if error_code == "ModelNotReadyException":
                raise RuntimeError(
                    f"Model {self.model_id} not ready: {error_msg}"
                ) from e
            if error_code == "ModelErrorException":
                raise RuntimeError(f"Model error: {error_msg}") from e
            if error_code == "InternalServerException":
                raise RuntimeError(f"Internal server error: {error_msg}") from e
            if error_code == "ServiceUnavailableException":
                raise RuntimeError(f"Service unavailable: {error_msg}") from e

            raise RuntimeError(f"Bedrock API error ({error_code}): {error_msg}") from e

        except BotoCoreError as e:
            error_msg = f"Bedrock connection error: {e}"
            logging.error(error_msg)
            raise ConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error in Bedrock converse call: {e}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def _ahandle_streaming_converse(
        self,
        messages: list[LLMMessage],
        body: BedrockConverseRequestBody,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        """Handle async streaming converse API call."""
        if response_model:
            # Check if structured_output tool already exists (from a previous recursive call)
            existing_tool_config = body.get("toolConfig")
            existing_tools: list[Any] = []
            structured_output_already_exists = False

            if existing_tool_config:
                existing_tools = list(existing_tool_config.get("tools", []))
                # Check if structured_output tool is already in the tools list
                for tool in existing_tools:
                    tool_spec = tool.get("toolSpec", {})
                    if tool_spec.get("name") == STRUCTURED_OUTPUT_TOOL_NAME:
                        structured_output_already_exists = True
                        break

            if not structured_output_already_exists:
                structured_tool: ConverseToolTypeDef = {
                    "toolSpec": {
                        "name": STRUCTURED_OUTPUT_TOOL_NAME,
                        "description": (
                            "Use this tool to provide your final structured response. "
                            "Call this tool when you have gathered all necessary information "
                            "and are ready to provide the final answer in the required format."
                        ),
                        "inputSchema": {
                            "json": generate_model_description(response_model)
                            .get("json_schema", {})
                            .get("schema", {})
                        },
                    }
                }

                if existing_tools:
                    # Append structured_output to existing tools, don't force toolChoice
                    existing_tools.append(structured_tool)
                    body["toolConfig"] = cast(
                        "ToolConfigurationTypeDef",
                        cast(object, {"tools": existing_tools}),
                    )
                else:
                    # No existing tools, use only structured_output with forced toolChoice
                    body["toolConfig"] = cast(
                        "ToolConfigurationTypeDef",
                        cast(
                            object,
                            {
                                "tools": [structured_tool],
                                "toolChoice": {
                                    "tool": {"name": STRUCTURED_OUTPUT_TOOL_NAME}
                                },
                            },
                        ),
                    )

        full_response = ""
        current_tool_use: dict[str, Any] | None = None
        tool_use_id: str | None = None
        tool_use_index = 0
        accumulated_tool_input = ""

        try:
            async_client = await self._ensure_async_client()
            response = await async_client.converse_stream(
                modelId=self.model_id,
                messages=cast(
                    "Sequence[MessageTypeDef | MessageOutputTypeDef]",
                    cast(object, messages),
                ),
                **body,
            )

            stream = response.get("stream")
            response_id = None
            if stream:
                async for event in stream:
                    if "messageStart" in event:
                        role = event["messageStart"].get("role")
                        logging.debug(f"Streaming message started with role: {role}")

                    elif "contentBlockStart" in event:
                        start = event["contentBlockStart"].get("start", {})
                        content_block_index = event["contentBlockStart"].get(
                            "contentBlockIndex", 0
                        )
                        if "toolUse" in start:
                            tool_use_block = start["toolUse"]
                            current_tool_use = cast(dict[str, Any], tool_use_block)
                            tool_use_id = current_tool_use.get("toolUseId")
                            tool_use_index = content_block_index
                            accumulated_tool_input = ""
                            self._emit_stream_chunk_event(
                                chunk="",
                                from_task=from_task,
                                from_agent=from_agent,
                                tool_call={
                                    "id": tool_use_id or "",
                                    "function": {
                                        "name": current_tool_use.get("name", ""),
                                        "arguments": "",
                                    },
                                    "type": "function",
                                    "index": tool_use_index,
                                },
                                call_type=LLMCallType.TOOL_CALL,
                                response_id=response_id,
                            )
                            logging.debug(
                                f"Tool use started in stream: {current_tool_use.get('name')} (ID: {tool_use_id})"
                            )

                    elif "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"]["delta"]
                        if "text" in delta:
                            text_chunk = delta["text"]
                            logging.debug(f"Streaming text chunk: {text_chunk[:50]}...")
                            full_response += text_chunk
                            self._emit_stream_chunk_event(
                                chunk=text_chunk,
                                from_task=from_task,
                                from_agent=from_agent,
                                response_id=response_id,
                            )
                        elif "toolUse" in delta and current_tool_use:
                            tool_input = delta["toolUse"].get("input", "")
                            if tool_input:
                                accumulated_tool_input += tool_input
                                logging.debug(f"Tool input delta: {tool_input}")
                                self._emit_stream_chunk_event(
                                    chunk=tool_input,
                                    from_task=from_task,
                                    from_agent=from_agent,
                                    tool_call={
                                        "id": tool_use_id or "",
                                        "function": {
                                            "name": current_tool_use.get("name", ""),
                                            "arguments": accumulated_tool_input,
                                        },
                                        "type": "function",
                                        "index": tool_use_index,
                                    },
                                    call_type=LLMCallType.TOOL_CALL,
                                    response_id=response_id,
                                )

                    elif "contentBlockStop" in event:
                        logging.debug("Content block stopped in stream")
                        if current_tool_use:
                            function_name = current_tool_use["name"]
                            function_args = cast(
                                dict[str, Any], current_tool_use.get("input", {})
                            )

                            # Check if this is the structured_output tool
                            if (
                                function_name == STRUCTURED_OUTPUT_TOOL_NAME
                                and response_model
                            ):
                                function_args = _preprocess_structured_data(
                                    function_args, response_model
                                )
                                try:
                                    result = response_model.model_validate(
                                        function_args
                                    )
                                    self._emit_call_completed_event(
                                        response=result.model_dump_json(),
                                        call_type=LLMCallType.LLM_CALL,
                                        from_task=from_task,
                                        from_agent=from_agent,
                                        messages=messages,
                                    )
                                    return result  # type: ignore[return-value]
                                except Exception as e:
                                    error_msg = (
                                        f"Failed to validate {STRUCTURED_OUTPUT_TOOL_NAME} tool response "
                                        f"with model {response_model.__name__}: {e}"
                                    )
                                    logging.error(error_msg)
                                    raise ValueError(error_msg) from e

                            # Handle regular tool execution
                            if available_functions:
                                tool_result = self._handle_tool_execution(
                                    function_name=function_name,
                                    function_args=function_args,
                                    available_functions=available_functions,
                                    from_task=from_task,
                                    from_agent=from_agent,
                                )

                                if tool_result is not None and tool_use_id:
                                    messages.append(
                                        {
                                            "role": "assistant",
                                            "content": [{"toolUse": current_tool_use}],
                                        }
                                    )

                                    messages.append(
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "toolResult": {
                                                        "toolUseId": tool_use_id,
                                                        "content": [
                                                            {"text": str(tool_result)}
                                                        ],
                                                    }
                                                }
                                            ],
                                        }
                                    )

                                    return await self._ahandle_converse(
                                        messages,
                                        body,
                                        available_functions,
                                        from_task,
                                        from_agent,
                                        response_model,
                                    )
                            current_tool_use = None
                            tool_use_id = None

                    elif "messageStop" in event:
                        stop_reason = event["messageStop"].get("stopReason")
                        logging.debug(f"Streaming message stopped: {stop_reason}")
                        if stop_reason == "max_tokens":
                            logging.warning(
                                "Streaming response truncated due to max_tokens"
                            )
                        elif stop_reason == "content_filtered":
                            logging.warning(
                                "Streaming response filtered due to content policy"
                            )
                        break

                    elif "metadata" in event:
                        metadata = event["metadata"]
                        if "usage" in metadata:
                            usage_metrics = metadata["usage"]
                            self._track_token_usage_internal(usage_metrics)
                            logging.debug(f"Token usage: {usage_metrics}")
                        if "trace" in metadata:
                            logging.debug(
                                f"Trace information available: {metadata['trace']}"
                            )

        except ClientError as e:
            error_msg = self._handle_client_error(e)
            raise RuntimeError(error_msg) from e
        except BotoCoreError as e:
            error_msg = f"Bedrock streaming connection error: {e}"
            logging.error(error_msg)
            raise ConnectionError(error_msg) from e

        full_response = self._apply_stop_words(full_response)

        if not full_response or full_response.strip() == "":
            logging.warning("Bedrock streaming returned empty content, using fallback")
            full_response = (
                "I apologize, but I couldn't generate a response. Please try again."
            )

        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=messages,
        )

        return self._invoke_after_llm_call_hooks(
            messages,
            full_response,
            from_agent,
        )

    def _format_messages_for_converse(
        self, messages: str | list[LLMMessage]
    ) -> tuple[list[LLMMessage], str | None]:
        """Format messages for Converse API following AWS documentation.

        Note: Returns dict[str, Any] instead of LLMMessage because Bedrock uses
        a different content structure: {"role": str, "content": [{"text": str}]}
        rather than the standard {"role": str, "content": str}.
        """
        # Use base class formatting first
        formatted_messages = self._format_messages(messages)

        converse_messages: list[LLMMessage] = []
        system_message: str | None = None

        for message in formatted_messages:
            role = message.get("role")
            content = message.get("content", "")
            tool_calls = message.get("tool_calls")
            tool_call_id = message.get("tool_call_id")

            if role == "system":
                # Extract system message - Converse API handles it separately
                if system_message:
                    system_message += f"\n\n{content}"
                else:
                    system_message = cast(str, content)
            elif role == "assistant" and tool_calls:
                # Convert OpenAI-style tool_calls to Bedrock toolUse format
                bedrock_content = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_use_block = {
                        "toolUse": {
                            "toolUseId": tc.get("id", f"call_{id(tc)}"),
                            "name": func.get("name", ""),
                            "input": func.get("arguments", {})
                            if isinstance(func.get("arguments"), dict)
                            else json.loads(func.get("arguments", "{}") or "{}"),
                        }
                    }
                    bedrock_content.append(tool_use_block)
                converse_messages.append(
                    {"role": "assistant", "content": bedrock_content}
                )
            elif role == "tool":
                if not tool_call_id:
                    raise ValueError("Tool message missing required tool_call_id")
                converse_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": tool_call_id,
                                    "content": [
                                        {"text": str(content) if content else ""}
                                    ],
                                }
                            }
                        ],
                    }
                )
            else:
                # Convert to Converse API format with proper content structure
                if isinstance(content, list):
                    # Already formatted as multimodal content blocks
                    converse_messages.append({"role": role, "content": content})
                else:
                    # String content - wrap in text block
                    text_content = content if content else ""
                    converse_messages.append(
                        {"role": role, "content": [{"text": text_content}]}
                    )

        # CRITICAL: Handle model-specific conversation requirements
        # Cohere and some other models require conversation to end with user message
        if converse_messages:
            last_message = converse_messages[-1]
            if last_message["role"] == "assistant":
                # For Cohere models, add a continuation user message
                if "cohere" in self.model.lower():
                    converse_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "text": "Please continue and provide your final answer."
                                }
                            ],
                        }
                    )
                # For other models that might have similar requirements
                elif any(
                    model_family in self.model.lower()
                    for model_family in ["command", "coral"]
                ):
                    converse_messages.append(
                        {
                            "role": "user",
                            "content": [{"text": "Continue your response."}],
                        }
                    )

        # Ensure first message is from user (required by Converse API)
        if not converse_messages:
            converse_messages.append(
                {
                    "role": "user",
                    "content": [{"text": "Hello, please help me with my request."}],
                }
            )
        elif converse_messages[0]["role"] != "user":
            converse_messages.insert(
                0,
                {
                    "role": "user",
                    "content": [{"text": "Hello, please help me with my request."}],
                },
            )

        return converse_messages, system_message

    @staticmethod
    def _messages_contain_tool_content(messages: list[LLMMessage]) -> bool:
        """Check if messages contain toolUse or toolResult content blocks.

        Bedrock requires toolConfig when messages have tool-related content.
        """
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if "toolUse" in block or "toolResult" in block:
                            return True
        return False

    @staticmethod
    def _extract_tools_from_message_history(
        messages: list[LLMMessage],
    ) -> list[dict[str, Any]]:
        """Extract tool definitions from toolUse blocks in message history.

        When no tools are passed but messages contain toolUse, we need to
        recreate a minimal toolConfig to satisfy Bedrock's API requirements.
        """
        tools: list[dict[str, Any]] = []
        seen_tool_names: set[str] = set()

        for message in messages:
            content = message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "toolUse" in block:
                        tool_use = block["toolUse"]
                        tool_name = tool_use.get("name", "")
                        if tool_name and tool_name not in seen_tool_names:
                            seen_tool_names.add(tool_name)
                            # Create a minimal tool spec from the toolUse block
                            tool_spec: dict[str, Any] = {
                                "toolSpec": {
                                    "name": tool_name,
                                    "description": f"Tool: {tool_name}",
                                    "inputSchema": {
                                        "json": {
                                            "type": "object",
                                            "properties": {},
                                        }
                                    },
                                }
                            }
                            tools.append(tool_spec)
        return tools

    @staticmethod
    def _format_tools_for_converse(
        tools: list[dict[str, Any]],
    ) -> list[ConverseToolTypeDef]:
        """Convert CrewAI tools to Converse API format following AWS specification."""
        from crewai.llms.providers.utils.common import safe_tool_conversion

        converse_tools: list[ConverseToolTypeDef] = []

        for tool in tools:
            try:
                name, description, parameters = safe_tool_conversion(tool, "Bedrock")

                tool_spec: ToolSpec = {
                    "name": name,
                    "description": description,
                }

                if parameters and isinstance(parameters, dict):
                    input_schema: ToolInputSchema = {"json": parameters}
                    tool_spec["inputSchema"] = input_schema

                converse_tool: ConverseToolTypeDef = {"toolSpec": tool_spec}

                converse_tools.append(converse_tool)

            except Exception as e:  # noqa: PERF203
                logging.warning(
                    f"Failed to convert tool {tool.get('name', 'unknown')}: {e}"
                )
                continue

        return converse_tools

    def _get_inference_config(self) -> EnhancedInferenceConfigurationTypeDef:
        """Get inference configuration following AWS Converse API specification."""
        config: EnhancedInferenceConfigurationTypeDef = {}

        if self.max_tokens:
            config["maxTokens"] = self.max_tokens

        if self.temperature is not None:
            config["temperature"] = float(self.temperature)
        if self.top_p is not None:
            config["topP"] = float(self.top_p)
        if self.stop_sequences:
            config["stopSequences"] = self.stop_sequences

        if self.is_claude_model and self.top_k is not None:
            # top_k is supported by Claude models
            config["topK"] = int(self.top_k)

        return config

    def _handle_client_error(self, e: ClientError) -> str:
        """Handle AWS ClientError with specific error codes and return error message."""
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_msg = e.response.get("Error", {}).get("Message", str(e))

        error_mapping = {
            "AccessDeniedException": f"Access denied to model {self.model_id}: {error_msg}",
            "ResourceNotFoundException": f"Model {self.model_id} not found: {error_msg}",
            "ThrottlingException": f"API throttled, please retry later: {error_msg}",
            "ValidationException": f"Invalid request: {error_msg}",
            "ModelTimeoutException": f"Model request timed out: {error_msg}",
            "ServiceQuotaExceededException": f"Service quota exceeded: {error_msg}",
            "ModelNotReadyException": f"Model {self.model_id} not ready: {error_msg}",
            "ModelErrorException": f"Model error: {error_msg}",
        }

        full_error_msg = error_mapping.get(
            error_code, f"Bedrock API error: {error_msg}"
        )
        logging.error(f"Bedrock client error ({error_code}): {full_error_msg}")

        return full_error_msg

    def _track_token_usage_internal(self, usage: TokenUsageTypeDef) -> None:  # type: ignore[override]
        """Track token usage from Bedrock response."""
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        total_tokens = usage.get("totalTokens", input_tokens + output_tokens)

        self._token_usage["prompt_tokens"] += input_tokens
        self._token_usage["completion_tokens"] += output_tokens
        self._token_usage["total_tokens"] += total_tokens
        self._token_usage["successful_requests"] += 1

    def supports_function_calling(self) -> bool:
        """Check if the model supports function calling."""
        return self.supports_tools

    def supports_stop_words(self) -> bool:
        """Check if the model supports stop words."""
        return True

    def get_context_window_size(self) -> int:
        """Get the context window size for the model."""
        from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO

        # Context window sizes for common Bedrock models
        context_windows = {
            "anthropic.claude-3-5-sonnet": 200000,
            "anthropic.claude-3-5-haiku": 200000,
            "anthropic.claude-3-opus": 200000,
            "anthropic.claude-3-sonnet": 200000,
            "anthropic.claude-3-haiku": 200000,
            "anthropic.claude-3-7-sonnet": 200000,
            "anthropic.claude-v2": 100000,
            "amazon.titan-text-express": 8000,
            "ai21.j2-ultra": 8192,
            "cohere.command-text": 4096,
            "meta.llama2-13b-chat": 4096,
            "meta.llama2-70b-chat": 4096,
            "meta.llama3-70b-instruct": 128000,
            "deepseek.r1": 32768,
        }

        # Find the best match for the model name
        for model_prefix, size in context_windows.items():
            if self.model.startswith(model_prefix):
                return int(size * CONTEXT_WINDOW_USAGE_RATIO)

        # Default context window size
        return int(8192 * CONTEXT_WINDOW_USAGE_RATIO)

    def supports_multimodal(self) -> bool:
        """Check if the model supports multimodal inputs.

        Claude 3+ and Nova Lite/Pro/Premier on Bedrock support vision.

        Returns:
            True if the model supports images.
        """
        model_lower = self.model.lower()
        vision_models = (
            "anthropic.claude-3",
            "amazon.nova-lite",
            "amazon.nova-pro",
            "amazon.nova-premier",
            "us.amazon.nova-lite",
            "us.amazon.nova-pro",
            "us.amazon.nova-premier",
        )
        return any(model_lower.startswith(m) for m in vision_models)

    def _is_nova_model(self) -> bool:
        """Check if the model is an Amazon Nova model.

        Only Nova models support S3 links for multimedia.

        Returns:
            True if the model is a Nova model.
        """
        model_lower = self.model.lower()
        return "amazon.nova-" in model_lower

    def get_file_uploader(self) -> Any:
        """Get a Bedrock S3 file uploader using this LLM's AWS credentials.

        Creates an S3 client using the same AWS credentials configured for
        this Bedrock LLM instance.

        Returns:
            BedrockFileUploader instance with pre-configured S3 client,
            or None if crewai_files is not installed.
        """
        try:
            import boto3
            from crewai_files.uploaders.bedrock import BedrockFileUploader

            s3_client = boto3.client(
                "s3",
                region_name=self.region_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
            )
            return BedrockFileUploader(
                region=self.region_name,
                client=s3_client,
            )
        except ImportError:
            return None

    def _get_document_format(self, content_type: str) -> str | None:
        """Map content type to Bedrock document format.

        Args:
            content_type: MIME type of the document.

        Returns:
            Bedrock format string or None if unsupported.
        """
        format_map = {
            "application/pdf": "pdf",
            "text/csv": "csv",
            "text/plain": "txt",
            "text/markdown": "md",
            "text/html": "html",
            "application/msword": "doc",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/vnd.ms-excel": "xls",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        }
        return format_map.get(content_type)

    def _get_video_format(self, content_type: str) -> str | None:
        """Map content type to Bedrock video format.

        Args:
            content_type: MIME type of the video.

        Returns:
            Bedrock format string or None if unsupported.
        """
        format_map = {
            "video/mp4": "mp4",
            "video/quicktime": "mov",
            "video/x-matroska": "mkv",
            "video/webm": "webm",
            "video/x-flv": "flv",
            "video/mpeg": "mpeg",
            "video/x-ms-wmv": "wmv",
            "video/3gpp": "three_gp",
        }
        return format_map.get(content_type)

    def format_text_content(self, text: str) -> dict[str, Any]:
        """Format text as a Bedrock content block.

        Bedrock uses {"text": "..."} format instead of {"type": "text", "text": "..."}.

        Args:
            text: The text content to format.

        Returns:
            A content block in Bedrock's expected format.
        """
        return {"text": text}
