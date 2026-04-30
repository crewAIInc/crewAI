from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
import json
import logging
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    TypedDict,
    cast,
)

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMCallType,
    LLMStreamChunkEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.llms.base_llm import (
    BaseLLM,
    JsonResponseFormat,
    get_current_call_id,
    llm_call_context,
)
from crewai.llms.constants import (
    ANTHROPIC_MODELS,
    AZURE_MODELS,
    BEDROCK_MODELS,
    GEMINI_MODELS,
    OPENAI_MODELS,
)
from crewai.utilities import InternalInstructor
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.logger_utils import suppress_warnings
from crewai.utilities.string_utils import sanitize_tool_name
from crewai.utilities.token_counter_callback import TokenCalcHandler


try:
    from crewai_files import aformat_multimodal_content, format_multimodal_content

    HAS_CREWAI_FILES = True
except ImportError:
    HAS_CREWAI_FILES = False


if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.utilities.types import LLMMessage

try:
    import litellm
    from litellm.litellm_core_utils.get_supported_openai_params import (
        get_supported_openai_params,
    )
    from litellm.types.utils import (
        ChatCompletionDeltaToolCall,
        Choices,
        Delta as LiteLLMDelta,
        Function,
        Message,
        ModelResponse,
        ModelResponseBase,
        ModelResponseStream,
        StreamingChoices as LiteLLMStreamingChoices,
    )
    from litellm.utils import supports_response_schema

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None  # type: ignore[assignment]
    Choices = None  # type: ignore[assignment, misc]
    LiteLLMDelta = None  # type: ignore[assignment, misc]
    Message = None  # type: ignore[assignment, misc]
    ModelResponseBase = None  # type: ignore[assignment, misc]
    ModelResponseStream = None  # type: ignore[assignment, misc]
    LiteLLMStreamingChoices = None  # type: ignore[assignment, misc]
    get_supported_openai_params = None  # type: ignore[assignment]
    ChatCompletionDeltaToolCall = None  # type: ignore[assignment, misc]
    Function = None  # type: ignore[assignment, misc]
    ModelResponse = None  # type: ignore[assignment, misc]
    supports_response_schema = None  # type: ignore[assignment]


load_dotenv()
logger = logging.getLogger(__name__)
if LITELLM_AVAILABLE:
    litellm.suppress_debug_info = True


MIN_CONTEXT: Final[int] = 1024
MAX_CONTEXT: Final[int] = 2097152  # Current max from gemini-1.5-pro
ANTHROPIC_PREFIXES: Final[tuple[str, str, str]] = ("anthropic/", "claude-", "claude/")

LLM_CONTEXT_WINDOW_SIZES: Final[dict[str, int]] = {
    # openai
    "gpt-4": 8192,
    "gpt-4o": 128000,
    "gpt-4o-mini": 200000,
    "gpt-4-turbo": 128000,
    "gpt-4.1": 1047576,  # Based on official docs
    "gpt-4.1-mini-2025-04-14": 1047576,
    "gpt-4.1-nano-2025-04-14": 1047576,
    "o1-preview": 128000,
    "o1-mini": 128000,
    "o3-mini": 200000,
    "o4-mini": 200000,
    # gemini
    "gemini-3-pro-preview": 1048576,
    "gemini-2.0-flash": 1048576,
    "gemini-2.0-flash-thinking-exp-01-21": 32768,
    "gemini-2.0-flash-lite-001": 1048576,
    "gemini-2.0-flash-001": 1048576,
    "gemini-2.5-flash-preview-04-17": 1048576,
    "gemini-2.5-pro-exp-03-25": 1048576,
    "gemini-1.5-pro": 2097152,
    "gemini-1.5-flash": 1048576,
    "gemini-1.5-flash-8b": 1048576,
    "gemini/gemma-3-1b-it": 32000,
    "gemini/gemma-3-4b-it": 128000,
    "gemini/gemma-3-12b-it": 128000,
    "gemini/gemma-3-27b-it": 128000,
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
    # bedrock
    "us.amazon.nova-pro-v1:0": 300000,
    "us.amazon.nova-micro-v1:0": 128000,
    "us.amazon.nova-lite-v1:0": 300000,
    # Claude 4 models
    "us.anthropic.claude-opus-4-7": 1000000,
    "us.anthropic.claude-sonnet-4-6": 1000000,
    "us.anthropic.claude-opus-4-6-v1": 1000000,
    "us.anthropic.claude-opus-4-5-20251101-v1:0": 200000,
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": 200000,
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": 200000,
    "us.anthropic.claude-opus-4-1-20250805-v1:0": 200000,
    "us.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": 200000,
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "us.anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "us.anthropic.claude-3-opus-20240229-v1:0": 200000,
    "us.anthropic.claude-3-haiku-20240307-v1:0": 200000,
    "us.meta.llama3-2-11b-instruct-v1:0": 128000,
    "us.meta.llama3-2-3b-instruct-v1:0": 131000,
    "us.meta.llama3-2-90b-instruct-v1:0": 128000,
    "us.meta.llama3-2-1b-instruct-v1:0": 131000,
    "us.meta.llama3-1-8b-instruct-v1:0": 128000,
    "us.meta.llama3-1-70b-instruct-v1:0": 128000,
    "us.meta.llama3-3-70b-instruct-v1:0": 128000,
    "us.meta.llama3-1-405b-instruct-v1:0": 128000,
    "eu.anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
    "eu.anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "eu.anthropic.claude-3-haiku-20240307-v1:0": 200000,
    # Claude 4 EU
    "eu.anthropic.claude-opus-4-7": 1000000,
    "eu.anthropic.claude-sonnet-4-6": 1000000,
    "eu.anthropic.claude-opus-4-6-v1": 1000000,
    "eu.anthropic.claude-opus-4-5-20251101-v1:0": 200000,
    "eu.anthropic.claude-haiku-4-5-20251001-v1:0": 200000,
    "eu.anthropic.claude-sonnet-4-5-20250929-v1:0": 200000,
    "eu.anthropic.claude-opus-4-1-20250805-v1:0": 200000,
    "eu.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "eu.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "eu.meta.llama3-2-3b-instruct-v1:0": 131000,
    "eu.meta.llama3-2-1b-instruct-v1:0": 131000,
    "apac.anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
    "apac.anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
    "apac.anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "apac.anthropic.claude-3-haiku-20240307-v1:0": 200000,
    # Claude 4 APAC
    "apac.anthropic.claude-opus-4-7": 1000000,
    "apac.anthropic.claude-sonnet-4-6": 1000000,
    "apac.anthropic.claude-opus-4-6-v1": 1000000,
    "apac.anthropic.claude-opus-4-5-20251101-v1:0": 200000,
    "apac.anthropic.claude-haiku-4-5-20251001-v1:0": 200000,
    "apac.anthropic.claude-sonnet-4-5-20250929-v1:0": 200000,
    "apac.anthropic.claude-opus-4-1-20250805-v1:0": 200000,
    "apac.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "apac.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "amazon.nova-pro-v1:0": 300000,
    "amazon.nova-micro-v1:0": 128000,
    "amazon.nova-lite-v1:0": 300000,
    "anthropic.claude-opus-4-7": 1000000,
    "anthropic.claude-sonnet-4-6": 1000000,
    "anthropic.claude-opus-4-6-v1": 1000000,
    "anthropic.claude-opus-4-5-20251101-v1:0": 200000,
    "anthropic.claude-haiku-4-5-20251001-v1:0": 200000,
    "anthropic.claude-sonnet-4-5-20250929-v1:0": 200000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
    "anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
    "anthropic.claude-3-5-haiku-20241022-v1:0": 200000,
    "anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
    "anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "anthropic.claude-3-opus-20240229-v1:0": 200000,
    "anthropic.claude-3-haiku-20240307-v1:0": 200000,
    "anthropic.claude-v2:1": 200000,
    "anthropic.claude-v2": 100000,
    "anthropic.claude-instant-v1": 100000,
    "meta.llama3-1-405b-instruct-v1:0": 128000,
    "meta.llama3-1-70b-instruct-v1:0": 128000,
    "meta.llama3-1-8b-instruct-v1:0": 128000,
    "meta.llama3-70b-instruct-v1:0": 8000,
    "meta.llama3-8b-instruct-v1:0": 8000,
    "amazon.titan-text-lite-v1": 4000,
    "amazon.titan-text-express-v1": 8000,
    "cohere.command-text-v14": 4000,
    "ai21.j2-mid-v1": 8191,
    "ai21.j2-ultra-v1": 8191,
    "ai21.jamba-instruct-v1:0": 256000,
    "mistral.mistral-7b-instruct-v0:2": 32000,
    "mistral.mixtral-8x7b-instruct-v0:1": 32000,
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

DEFAULT_CONTEXT_WINDOW_SIZE: Final[int] = 8192
CONTEXT_WINDOW_USAGE_RATIO: Final[float] = 0.85
SUPPORTED_NATIVE_PROVIDERS: Final[list[str]] = [
    "openai",
    "anthropic",
    "claude",
    "azure",
    "azure_openai",
    "google",
    "gemini",
    "bedrock",
    "aws",
    # OpenAI-compatible providers
    "openrouter",
    "deepseek",
    "ollama",
    "ollama_chat",
    "hosted_vllm",
    "cerebras",
    "dashscope",
]


class Delta(TypedDict):
    content: str | None
    role: str | None


class StreamingChoices(TypedDict):
    delta: Delta
    index: int
    finish_reason: str | None


class FunctionArgs(BaseModel):
    name: str = ""
    arguments: str = ""


class AccumulatedToolArgs(BaseModel):
    function: FunctionArgs = Field(default_factory=FunctionArgs)


class LLM(BaseLLM):
    llm_type: Literal["litellm"] = "litellm"
    completion_cost: float | None = None
    timeout: float | int | None = None
    top_p: float | None = None
    n: int | None = None
    max_completion_tokens: int | None = None
    max_tokens: int | float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[int, float] | None = None
    response_format: JsonResponseFormat | type[BaseModel] | None = None
    seed: int | None = None
    logprobs: int | None = None
    top_logprobs: int | None = None
    api_base: str | None = None
    api_version: str | None = None
    callbacks: list[Any] | None = None
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    stream: bool = False
    interceptor: Any = None
    thinking: Any = None
    context_window_size: int = 0
    is_anthropic: bool = False

    def __new__(cls, model: str, is_litellm: bool = False, **kwargs: Any) -> LLM:
        """Factory method that routes to native SDK or falls back to LiteLLM.

        Routing priority:
            1. If 'provider' kwarg is present, use that provider with constants
            2. If only 'model' kwarg, use constants to infer provider
            3. If "/" in model name:
               - Check if prefix is a native provider (openai/anthropic/azure/bedrock/gemini)
               - If yes, validate model against constants
               - If valid, route to native SDK; otherwise route to LiteLLM
        """
        if not model or not isinstance(model, str):
            raise ValueError("Model must be a non-empty string")

        explicit_provider = kwargs.get("provider")

        if explicit_provider:
            provider = explicit_provider
            use_native = True
            model_string = model
        elif "/" in model:
            prefix, _, model_part = model.partition("/")

            provider_mapping = {
                "openai": "openai",
                "anthropic": "anthropic",
                "claude": "anthropic",
                "azure": "azure",
                "azure_openai": "azure",
                "google": "gemini",
                "gemini": "gemini",
                "bedrock": "bedrock",
                "aws": "bedrock",
                # OpenAI-compatible providers
                "openrouter": "openrouter",
                "deepseek": "deepseek",
                "ollama": "ollama",
                "ollama_chat": "ollama_chat",
                "hosted_vllm": "hosted_vllm",
                "cerebras": "cerebras",
                "dashscope": "dashscope",
            }

            canonical_provider = provider_mapping.get(prefix.lower())

            if canonical_provider and cls._validate_model_in_constants(
                model_part, canonical_provider
            ):
                provider = canonical_provider
                use_native = True
                model_string = model_part
            else:
                provider = prefix
                use_native = False
                model_string = model_part
        else:
            provider = cls._infer_provider_from_model(model)
            use_native = True
            model_string = model

        native_class = cls._get_native_provider(provider) if use_native else None
        if native_class and not is_litellm and provider in SUPPORTED_NATIVE_PROVIDERS:
            try:
                # Remove 'provider' from kwargs if it exists to avoid duplicate keyword argument
                kwargs_copy = {k: v for k, v in kwargs.items() if k != "provider"}
                return cast(
                    Self,
                    native_class(model=model_string, provider=provider, **kwargs_copy),
                )
            except NotImplementedError:
                raise
            except Exception as e:
                raise ImportError(f"Error importing native provider: {e}") from e

        # FALLBACK to LiteLLM
        if not LITELLM_AVAILABLE:
            native_list = ", ".join(SUPPORTED_NATIVE_PROVIDERS)
            error_msg = (
                f"Unable to initialize LLM with model '{model}'. "
                f"The model did not match any supported native provider "
                f"({native_list}), and the LiteLLM fallback package is not "
                f"installed.\n\n"
                f"To fix this, either:\n"
                f"  1. Install LiteLLM for broad model support: "
                f"uv add 'crewai[litellm]'\n"
                f"or\n"
                f"pip install litellm\n\n"
                f"For more details, see: "
                f"https://docs.crewai.com/en/learn/llm-connections"
            )
            logger.error(error_msg)
            raise ImportError(error_msg) from None

        return object.__new__(cls)

    @classmethod
    def _matches_provider_pattern(cls, model: str, provider: str) -> bool:
        """Check if a model name matches provider-specific patterns.

        This allows supporting models that aren't in the hardcoded constants list,
        including "latest" versions and new models that follow provider naming conventions.

        Args:
            model: The model name to check
            provider: The provider to check against (canonical name)

        Returns:
            True if the model matches the provider's naming pattern, False otherwise
        """
        model_lower = model.lower()

        if provider == "openai":
            return any(
                model_lower.startswith(prefix)
                for prefix in ["gpt-", "o1", "o3", "o4", "whisper-"]
            )

        if provider == "anthropic" or provider == "claude":
            return any(
                model_lower.startswith(prefix) for prefix in ["claude-", "anthropic."]
            )

        if provider == "gemini" or provider == "google":
            return any(
                model_lower.startswith(prefix)
                for prefix in ["gemini-", "gemma-", "learnlm-"]
            )

        if provider == "bedrock":
            return "." in model_lower

        if provider == "azure":
            return any(
                model_lower.startswith(prefix)
                for prefix in ["gpt-", "gpt-35-", "o1", "o3", "o4", "azure-"]
            )

        # OpenAI-compatible providers - most accept any model name, but some
        # (DeepSeek, Dashscope) restrict to their own model prefixes
        if provider == "deepseek":
            return model_lower.startswith("deepseek")

        if provider == "ollama" or provider == "ollama_chat":
            # Ollama accepts any local model name
            return True

        if provider == "hosted_vllm":
            # vLLM serves any model
            return True

        if provider == "cerebras":
            return True

        if provider == "dashscope":
            return model_lower.startswith("qwen")

        if provider == "openrouter":
            # OpenRouter uses org/model format but accepts anything
            return True

        return False

    @classmethod
    def _validate_model_in_constants(cls, model: str, provider: str) -> bool:
        """Validate if a model name exists in the provider's constants or matches provider patterns.

        This method first checks the hardcoded constants list for known models.
        If not found, it falls back to pattern matching to support new models,
        "latest" versions, and models that follow provider naming conventions.

        Args:
            model: The model name to validate
            provider: The provider to check against (canonical name)

        Returns:
            True if the model exists in constants or matches provider patterns, False otherwise
        """
        if provider == "openai" and model in OPENAI_MODELS:
            return True

        if (
            provider == "anthropic" or provider == "claude"
        ) and model in ANTHROPIC_MODELS:
            return True

        if (provider == "gemini" or provider == "google") and model in GEMINI_MODELS:
            return True

        if provider == "bedrock" and model in BEDROCK_MODELS:
            return True

        if provider == "azure":
            # azure does not provide a list of available models, determine a better way to handle this
            return True

        # Fallback to pattern matching for models not in constants
        return cls._matches_provider_pattern(model, provider)

    @classmethod
    def _infer_provider_from_model(cls, model: str) -> str:
        """Infer the provider from the model name.

        This method first checks the hardcoded constants list for known models.
        If not found, it uses pattern matching to infer the provider from model name patterns.
        This allows supporting new models and "latest" versions without hardcoding.

        Args:
            model: The model name without provider prefix

        Returns:
            The inferred provider name, defaults to "openai"
        """
        if model in OPENAI_MODELS:
            return "openai"

        if model in ANTHROPIC_MODELS:
            return "anthropic"

        if model in GEMINI_MODELS:
            return "gemini"

        if model in BEDROCK_MODELS:
            return "bedrock"

        if model in AZURE_MODELS:
            return "azure"

        return "openai"

    @classmethod
    def _get_native_provider(cls, provider: str) -> type | None:
        """Get native provider class if available."""
        if provider == "openai":
            from crewai.llms.providers.openai.completion import OpenAICompletion

            return OpenAICompletion

        if provider == "anthropic" or provider == "claude":
            from crewai.llms.providers.anthropic.completion import (
                AnthropicCompletion,
            )

            return AnthropicCompletion

        if provider == "azure" or provider == "azure_openai":
            from crewai.llms.providers.azure.completion import AzureCompletion

            return AzureCompletion

        if provider == "google" or provider == "gemini":
            from crewai.llms.providers.gemini.completion import GeminiCompletion

            return GeminiCompletion

        if provider == "bedrock":
            from crewai.llms.providers.bedrock.completion import BedrockCompletion

            return BedrockCompletion

        # OpenAI-compatible providers
        openai_compatible_providers = {
            "openrouter",
            "deepseek",
            "ollama",
            "ollama_chat",
            "hosted_vllm",
            "cerebras",
            "dashscope",
        }
        if provider in openai_compatible_providers:
            from crewai.llms.providers.openai_compatible.completion import (
                OpenAICompatibleCompletion,
            )

            return OpenAICompatibleCompletion

        return None

    @model_validator(mode="before")
    @classmethod
    def _validate_llm_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        model = data.get("model", "")
        data["is_anthropic"] = cls._is_anthropic_model(model)
        return data

    @model_validator(mode="after")
    def _init_litellm(self) -> LLM:
        self.is_litellm = True
        if LITELLM_AVAILABLE:
            litellm.drop_params = True
            self.set_callbacks(self.callbacks or [])
            self.set_env_callbacks()
        return self

    @staticmethod
    def _is_anthropic_model(model: str) -> bool:
        """Determine if the model is from Anthropic provider.

        Args:
            model: The model identifier string.

        Returns:
            bool: True if the model is from Anthropic, False otherwise.
        """
        anthropic_prefixes = ("anthropic/", "claude-", "claude/")
        return any(prefix in model.lower() for prefix in anthropic_prefixes)

    def _prepare_completion_params(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        skip_file_processing: bool = False,
    ) -> dict[str, Any]:
        """Prepare parameters for the completion call.

        Args:
            messages: Input messages for the LLM
            tools: Optional list of tool schemas
            skip_file_processing: Skip file processing (used when already done async)

        Returns:
            Dict[str, Any]: Parameters for the completion call
        """
        # --- 1) Format messages according to provider requirements
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        # --- 1a) Process any file attachments into multimodal content
        if not skip_file_processing:
            messages = self._process_message_files(messages)
        formatted_messages = self._format_messages_for_provider(messages)

        # --- 2) Prepare the parameters for the completion call
        params = {
            "model": self.model,
            "messages": formatted_messages,
            "timeout": self.timeout,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stop": (self.stop or None) if self.supports_stop_words() else None,
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
        params: dict[str, Any],
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> Any:
        """Handle a streaming response from the LLM.

        Args:
            params: Parameters for the completion call
            callbacks: Optional list of callback functions
            available_functions: Dict of available functions
            from_task: Optional task object
            from_agent: Optional agent object
            response_model: Optional response model

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

        accumulated_tool_args: defaultdict[int, AccumulatedToolArgs] = defaultdict(
            AccumulatedToolArgs
        )

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
                response_id = None

                if isinstance(chunk, ModelResponseBase):
                    response_id = chunk.id

                # Safely extract content from various chunk formats
                try:
                    # Try to access choices safely
                    choices = None
                    if isinstance(chunk, dict) and "choices" in chunk:
                        choices = chunk["choices"]
                    elif isinstance(chunk, ModelResponseStream):
                        choices = chunk.choices

                    # Try to extract usage information if available
                    # NOTE: usage is a pydantic extra field on ModelResponseBase,
                    # so it must be accessed via model_extra.
                    if isinstance(chunk, dict) and "usage" in chunk:
                        usage_info = chunk["usage"]
                    elif isinstance(chunk, ModelResponseBase) and chunk.model_extra:
                        usage_info = chunk.model_extra.get("usage") or usage_info

                    if choices and len(choices) > 0:
                        choice = choices[0]

                        # Handle different delta formats
                        delta = None
                        if isinstance(choice, dict) and "delta" in choice:
                            delta = choice["delta"]
                        elif isinstance(choice, LiteLLMStreamingChoices):
                            delta = choice.delta

                        # Extract content from delta
                        if delta:
                            # Handle dict format
                            if isinstance(delta, dict):
                                if "content" in delta and delta["content"] is not None:
                                    chunk_content = delta["content"]
                            # Handle object format
                            elif isinstance(delta, LiteLLMDelta):
                                chunk_content = delta.content

                            # Handle case where content might be None or empty
                            if chunk_content is None and isinstance(delta, dict):
                                # Some models might send empty content chunks
                                chunk_content = ""

                            # Enable tool calls using streaming
                            if "tool_calls" in delta:
                                tool_calls = delta["tool_calls"]
                                if tool_calls:
                                    result = self._handle_streaming_tool_calls(
                                        tool_calls=tool_calls,
                                        accumulated_tool_args=accumulated_tool_args,
                                        available_functions=available_functions,
                                        from_task=from_task,
                                        from_agent=from_agent,
                                        response_id=response_id,
                                    )

                                    if result is not None:
                                        chunk_content = result

                except Exception as e:
                    logging.debug(f"Error extracting content from chunk: {e}")
                    logging.debug(f"Chunk format: {type(chunk)}, content: {chunk}")

                # Only add non-None content to the response
                if chunk_content is not None:
                    # Add the chunk content to the full response
                    full_response += chunk_content

                    crewai_event_bus.emit(
                        self,
                        event=LLMStreamChunkEvent(
                            chunk=chunk_content,
                            from_task=from_task,
                            from_agent=from_agent,
                            call_type=LLMCallType.LLM_CALL,
                            response_id=response_id,
                            call_id=get_current_call_id(),
                        ),
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
                    non_streaming_params,
                    callbacks,
                    available_functions,
                    from_task,
                    from_agent,
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
                        elif isinstance(last_chunk, ModelResponseStream):
                            choices = last_chunk.choices

                        if choices and len(choices) > 0:
                            choice = choices[0]

                            # Try to get content from message
                            message = None
                            if isinstance(choice, dict) and "message" in choice:
                                message = choice["message"]
                            elif isinstance(choice, Choices):
                                message = choice.message

                            if message:
                                content = None
                                if isinstance(message, dict) and "content" in message:
                                    content = message["content"]
                                elif isinstance(message, Message):
                                    content = message.content

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
            if not full_response.strip() and len(accumulated_tool_args) == 0:
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
                    elif isinstance(last_chunk, ModelResponseStream):
                        choices = last_chunk.choices

                    if choices and len(choices) > 0:
                        choice = choices[0]

                        delta = None
                        if isinstance(choice, dict) and "delta" in choice:
                            delta = choice["delta"]
                        elif isinstance(choice, LiteLLMStreamingChoices):
                            delta = choice.delta

                        if delta:
                            if isinstance(delta, dict) and "tool_calls" in delta:
                                tool_calls = delta["tool_calls"]
                            elif isinstance(delta, LiteLLMDelta):
                                tool_calls = delta.tool_calls
            except Exception as e:
                logging.debug(f"Error checking for tool calls: {e}")

            # Track token usage and log callbacks if available in streaming mode
            if usage_info:
                self._track_token_usage_internal(usage_info)
            self._handle_streaming_callbacks(callbacks, usage_info, last_chunk)

            if not tool_calls or not available_functions:
                if response_model and self.is_litellm:
                    instructor_instance = InternalInstructor(
                        content=full_response,
                        model=response_model,
                        llm=self,
                    )
                    result = instructor_instance.to_pydantic()
                    structured_response = result.model_dump_json()
                    usage_dict = self._usage_to_dict(usage_info)
                    self._handle_emit_call_events(
                        response=structured_response,
                        call_type=LLMCallType.LLM_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                        messages=params["messages"],
                        usage=usage_dict,
                    )
                    return structured_response

                usage_dict = self._usage_to_dict(usage_info)
                self._handle_emit_call_events(
                    response=full_response,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                    usage=usage_dict,
                )
                return full_response

            # --- 9) Handle tool calls if present
            tool_result = self._handle_tool_call(tool_calls, available_functions)
            if tool_result is not None:
                return tool_result

            # --- 10) Emit completion event and return response
            usage_dict = self._usage_to_dict(usage_info)
            self._handle_emit_call_events(
                response=full_response,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=params["messages"],
                usage=usage_dict,
            )
            return full_response

        except LLMContextLengthExceededError:
            # Re-raise our own context length error
            raise
        except Exception as e:
            # Check if this is a context window error and convert to our exception type
            error_msg = str(e)
            if LLMContextLengthExceededError._is_context_limit_error(error_msg):
                raise LLMContextLengthExceededError(error_msg) from e

            logging.error(f"Error in streaming response: {e!s}")
            if full_response.strip():
                logging.warning(f"Returning partial response despite error: {e!s}")
                self._handle_emit_call_events(
                    response=full_response,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                    usage=self._usage_to_dict(usage_info),
                )
                return full_response

            crewai_event_bus.emit(
                self,
                event=LLMCallFailedEvent(
                    error=str(e),
                    from_task=from_task,
                    from_agent=from_agent,
                    call_id=get_current_call_id(),
                ),
            )
            raise Exception(f"Failed to get streaming response: {e!s}") from e

    def _handle_streaming_tool_calls(
        self,
        tool_calls: list[ChatCompletionDeltaToolCall],
        accumulated_tool_args: defaultdict[int, AccumulatedToolArgs],
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
        response_id: str | None = None,
    ) -> Any:
        for tool_call in tool_calls:
            current_tool_accumulator = accumulated_tool_args[tool_call.index]

            if tool_call.function.name:
                current_tool_accumulator.function.name = tool_call.function.name

            if tool_call.function.arguments:
                current_tool_accumulator.function.arguments += (
                    tool_call.function.arguments
                )

            crewai_event_bus.emit(
                self,
                event=LLMStreamChunkEvent(
                    tool_call=tool_call.to_dict(),
                    chunk=tool_call.function.arguments,
                    from_task=from_task,
                    from_agent=from_agent,
                    call_type=LLMCallType.TOOL_CALL,
                    response_id=response_id,
                    call_id=get_current_call_id(),
                ),
            )

            if (
                current_tool_accumulator.function.name
                and current_tool_accumulator.function.arguments
                and available_functions
            ):
                try:
                    json.loads(current_tool_accumulator.function.arguments)

                    return self._handle_tool_call(
                        [current_tool_accumulator],
                        available_functions,
                    )
                except json.JSONDecodeError:
                    continue
        return None

    @staticmethod
    def _handle_streaming_callbacks(
        callbacks: list[Any] | None,
        usage_info: dict[str, Any] | None,
        last_chunk: Any | None,
    ) -> None:
        """Handle callbacks with usage info for streaming responses.

        Args:
            callbacks: Optional list of callback functions
            usage_info: Usage information collected during streaming
            last_chunk: The last chunk received from the streaming response
        """
        if callbacks and len(callbacks) > 0:
            for callback in callbacks:
                if isinstance(callback, TokenCalcHandler):
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
                                elif (
                                    isinstance(last_chunk, ModelResponseBase)
                                    and last_chunk.model_extra
                                ):
                                    usage_info = (
                                        last_chunk.model_extra.get("usage")
                                        or usage_info
                                    )
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
        params: dict[str, Any],
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle a non-streaming response from the LLM.

        Args:
            params: Parameters for the completion call
            callbacks: Optional list of callback functions
            available_functions: Dict of available functions
            from_task: Optional Task that invoked the LLM
            from_agent: Optional Agent that invoked the LLM
            response_model: Optional Response model

        Returns:
            str: The response text
        """
        # --- 1) Handle response_model with InternalInstructor for LiteLLM
        if response_model and self.is_litellm:
            from crewai.utilities.internal_instructor import InternalInstructor

            messages = params.get("messages", [])
            if not messages:
                raise ValueError("Messages are required when using response_model")

            # Combine all message content for InternalInstructor
            combined_content = "\n\n".join(
                f"{msg['role'].upper()}: {msg['content']}" for msg in messages
            )

            instructor_instance = InternalInstructor(
                content=combined_content,
                model=response_model,
                llm=self,
            )
            result = instructor_instance.to_pydantic()
            structured_response = result.model_dump_json()
            self._handle_emit_call_events(
                response=structured_response,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=messages,
                usage=None,
            )
            return structured_response

        try:
            # Attempt to make the completion call, but catch context window errors
            # and convert them to our own exception type for consistent handling
            # across the codebase. This allows CrewAgentExecutor to handle context
            # length issues appropriately.
            if response_model:
                params["response_model"] = response_model
            response = litellm.completion(**params)

            if isinstance(response, ModelResponseBase) and response.model_extra:
                usage_info = response.model_extra.get("usage")
                if usage_info:
                    self._track_token_usage_internal(usage_info)

        except LLMContextLengthExceededError:
            # Re-raise our own context length error
            raise
        except Exception as e:
            # Check if this is a context window error and convert to our exception type
            error_msg = str(e)
            if LLMContextLengthExceededError._is_context_limit_error(error_msg):
                raise LLMContextLengthExceededError(error_msg) from e
            raise

        response_usage = self._usage_to_dict(
            response.model_extra.get("usage")
            if isinstance(response, ModelResponseBase) and response.model_extra
            else None
        )

        # --- 2) Handle structured output response (when response_model is provided)
        if response_model is not None:
            # When using instructor/response_model, litellm returns a Pydantic model instance
            if isinstance(response, BaseModel):
                structured_response = response.model_dump_json()
                self._handle_emit_call_events(
                    response=structured_response,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                    usage=response_usage,
                )
                return structured_response

        # --- 3) Extract response message and content (standard response)
        response_message = cast(Choices, cast(ModelResponse, response).choices)[
            0
        ].message
        text_response = response_message.content or ""
        # --- 3) Handle callbacks with usage info
        if callbacks and len(callbacks) > 0:
            for callback in callbacks:
                if isinstance(callback, TokenCalcHandler):
                    usage_info = (
                        response.model_extra.get("usage")
                        if isinstance(response, ModelResponseBase)
                        and response.model_extra
                        else None
                    )
                    if usage_info:
                        callback.log_success_event(
                            kwargs=params,
                            response_obj={"usage": usage_info},
                            start_time=0,
                            end_time=0,
                        )
        # --- 4) Check for tool calls
        tool_calls = response_message.tool_calls or []

        # --- 5) If there are tool calls but no available functions, return the tool calls
        if tool_calls and not available_functions:
            return tool_calls

        # --- 6) If there are no tool calls to execute, return the text response directly
        if not tool_calls and text_response:
            self._handle_emit_call_events(
                response=text_response,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=params["messages"],
                usage=response_usage,
            )
            return text_response

        # --- 7) Handle tool calls if present (execute when available_functions provided)
        if tool_calls and available_functions:
            tool_result = self._handle_tool_call(
                tool_calls, available_functions, from_task, from_agent
            )
            if tool_result is not None:
                return tool_result

        # --- 8) If tool call handling didn't return a result, emit completion event and return text response
        self._handle_emit_call_events(
            response=text_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=params["messages"],
            usage=response_usage,
        )
        return text_response

    async def _ahandle_non_streaming_response(
        self,
        params: dict[str, Any],
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle an async non-streaming response from the LLM.

        Args:
            params: Parameters for the completion call
            callbacks: Optional list of callback functions
            available_functions: Dict of available functions
            from_task: Optional Task that invoked the LLM
            from_agent: Optional Agent that invoked the LLM
            response_model: Optional Response model

        Returns:
            str: The response text
        """
        if response_model and self.is_litellm:
            from crewai.utilities.internal_instructor import InternalInstructor

            messages = params.get("messages", [])
            if not messages:
                raise ValueError("Messages are required when using response_model")

            combined_content = "\n\n".join(
                f"{msg['role'].upper()}: {msg['content']}" for msg in messages
            )

            instructor_instance = InternalInstructor(
                content=combined_content,
                model=response_model,
                llm=self,
            )
            result = instructor_instance.to_pydantic()
            structured_response = result.model_dump_json()
            self._handle_emit_call_events(
                response=structured_response,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=messages,
                usage=None,
            )
            return structured_response

        try:
            if response_model:
                params["response_model"] = response_model
            response = await litellm.acompletion(**params)

            if isinstance(response, ModelResponseBase) and response.model_extra:
                usage_info = response.model_extra.get("usage")
                if usage_info:
                    self._track_token_usage_internal(usage_info)

        except LLMContextLengthExceededError:
            # Re-raise our own context length error
            raise
        except Exception as e:
            # Check if this is a context window error and convert to our exception type
            error_msg = str(e)
            if LLMContextLengthExceededError._is_context_limit_error(error_msg):
                raise LLMContextLengthExceededError(error_msg) from e
            raise

        response_usage = self._usage_to_dict(
            response.model_extra.get("usage")
            if isinstance(response, ModelResponseBase) and response.model_extra
            else None
        )

        if response_model is not None:
            if isinstance(response, BaseModel):
                structured_response = response.model_dump_json()
                self._handle_emit_call_events(
                    response=structured_response,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params["messages"],
                    usage=response_usage,
                )
                return structured_response

        response_message = cast(Choices, cast(ModelResponse, response).choices)[
            0
        ].message
        text_response = response_message.content or ""

        if callbacks and len(callbacks) > 0:
            for callback in callbacks:
                if isinstance(callback, TokenCalcHandler):
                    usage_info = (
                        response.model_extra.get("usage")
                        if isinstance(response, ModelResponseBase)
                        and response.model_extra
                        else None
                    )
                    if usage_info:
                        callback.log_success_event(
                            kwargs=params,
                            response_obj={"usage": usage_info},
                            start_time=0,
                            end_time=0,
                        )

        tool_calls = response_message.tool_calls or []

        if tool_calls and not available_functions:
            return tool_calls

        if not tool_calls and text_response:
            self._handle_emit_call_events(
                response=text_response,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=params["messages"],
                usage=response_usage,
            )
            return text_response

        # Handle tool calls if present (execute when available_functions provided)
        if tool_calls and available_functions:
            tool_result = self._handle_tool_call(
                tool_calls, available_functions, from_task, from_agent
            )
            if tool_result is not None:
                return tool_result

        self._handle_emit_call_events(
            response=text_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=params["messages"],
            usage=response_usage,
        )
        return text_response

    async def _ahandle_streaming_response(
        self,
        params: dict[str, Any],
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> Any:
        """Handle an async streaming response from the LLM.

        Args:
            params: Parameters for the completion call
            callbacks: Optional list of callback functions
            available_functions: Dict of available functions
            from_task: Optional task object
            from_agent: Optional agent object
            response_model: Optional response model

        Returns:
            str: The complete response text
        """
        full_response = ""
        chunk_count = 0

        usage_info = None

        accumulated_tool_args: defaultdict[int, AccumulatedToolArgs] = defaultdict(
            AccumulatedToolArgs
        )

        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
        response_id = None

        try:
            async for chunk in await litellm.acompletion(**params):
                chunk_count += 1
                chunk_content = None
                response_id = chunk.id if isinstance(chunk, ModelResponseBase) else None

                try:
                    choices = None
                    if isinstance(chunk, dict) and "choices" in chunk:
                        choices = chunk["choices"]
                    elif isinstance(chunk, ModelResponseStream):
                        choices = chunk.choices

                    if isinstance(chunk, ModelResponseBase) and chunk.model_extra:
                        chunk_usage = chunk.model_extra.get("usage")
                        if chunk_usage is not None:
                            usage_info = chunk_usage

                    if choices and len(choices) > 0:
                        first_choice = choices[0]
                        delta = None

                        if isinstance(first_choice, dict):
                            delta = first_choice.get("delta", {})
                        elif isinstance(first_choice, LiteLLMStreamingChoices):
                            delta = first_choice.delta

                        if delta:
                            if isinstance(delta, dict):
                                chunk_content = delta.get("content")
                            elif isinstance(delta, LiteLLMDelta):
                                chunk_content = delta.content

                            tool_calls: list[ChatCompletionDeltaToolCall] | None = None
                            if isinstance(delta, dict):
                                tool_calls = delta.get("tool_calls")
                            elif isinstance(delta, LiteLLMDelta):
                                tool_calls = delta.tool_calls

                            if tool_calls:
                                for tool_call in tool_calls:
                                    idx = tool_call.index
                                    if tool_call.function:
                                        if tool_call.function.name:
                                            accumulated_tool_args[
                                                idx
                                            ].function.name = tool_call.function.name
                                        if tool_call.function.arguments:
                                            accumulated_tool_args[
                                                idx
                                            ].function.arguments += (
                                                tool_call.function.arguments
                                            )

                except (AttributeError, KeyError, IndexError, TypeError):
                    pass

                if chunk_content:
                    full_response += chunk_content
                    crewai_event_bus.emit(
                        self,
                        event=LLMStreamChunkEvent(
                            chunk=chunk_content,
                            from_task=from_task,
                            from_agent=from_agent,
                            response_id=response_id,
                            call_id=get_current_call_id(),
                        ),
                    )

            if callbacks and len(callbacks) > 0 and usage_info:
                for callback in callbacks:
                    if isinstance(callback, TokenCalcHandler):
                        callback.log_success_event(
                            kwargs=params,
                            response_obj={"usage": usage_info},
                            start_time=0,
                            end_time=0,
                        )

            if usage_info:
                self._track_token_usage_internal(usage_info)

            if accumulated_tool_args and available_functions:
                # Convert accumulated tool args to ChatCompletionDeltaToolCall objects
                tool_calls_list: list[ChatCompletionDeltaToolCall] = [
                    ChatCompletionDeltaToolCall(
                        index=idx,
                        function=Function(
                            name=tool_arg.function.name,
                            arguments=tool_arg.function.arguments,
                        ),
                    )
                    for idx, tool_arg in accumulated_tool_args.items()
                    if tool_arg.function.name
                ]

                if tool_calls_list:
                    result = self._handle_streaming_tool_calls(
                        tool_calls=tool_calls_list,
                        accumulated_tool_args=accumulated_tool_args,
                        available_functions=available_functions,
                        from_task=from_task,
                        from_agent=from_agent,
                        response_id=response_id,
                    )
                    if result is not None:
                        return result

            usage_dict = self._usage_to_dict(usage_info)
            self._handle_emit_call_events(
                response=full_response,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=params.get("messages"),
                usage=usage_dict,
            )
            return full_response

        except LLMContextLengthExceededError:
            # Re-raise our own context length error
            raise
        except Exception as e:
            # Check if this is a context window error and convert to our exception type
            error_msg = str(e)
            if LLMContextLengthExceededError._is_context_limit_error(error_msg):
                raise LLMContextLengthExceededError(error_msg) from e

            if chunk_count == 0:
                raise
            if full_response:
                self._handle_emit_call_events(
                    response=full_response,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=params.get("messages"),
                    usage=self._usage_to_dict(usage_info),
                )
                return full_response
            raise

    def _handle_tool_call(
        self,
        tool_calls: list[Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
    ) -> Any:
        """Handle a tool call from the LLM.

        Args:
            tool_calls: List of tool calls from the LLM
            available_functions: Dict of available functions
            from_task: Optional Task that invoked the LLM
            from_agent: Optional Agent that invoked the LLM

        Returns:
            The result of the tool call, or None if no tool call was made
        """
        # --- 1) Validate tool calls and available functions
        if not tool_calls or not available_functions:
            return None

        # --- 2) Extract function name from first tool call
        tool_call = tool_calls[0]
        function_name = sanitize_tool_name(tool_call.function.name)
        function_args = {}  # Initialize to empty dict to avoid unbound variable

        # --- 3) Check if function is available
        if function_name in available_functions:
            try:
                # --- 3.1) Parse function arguments
                function_args = json.loads(tool_call.function.arguments)
                fn = available_functions[function_name]

                started_at = datetime.now()
                crewai_event_bus.emit(
                    self,
                    event=ToolUsageStartedEvent(
                        tool_name=function_name,
                        tool_args=function_args,
                        from_agent=from_agent,
                        from_task=from_task,
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
                        from_task=from_task,
                        from_agent=from_agent,
                    ),
                )

                # --- 3.3) Emit success event
                self._handle_emit_call_events(
                    response=result,
                    call_type=LLMCallType.TOOL_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                )
                return result
            except Exception as e:
                # --- 3.4) Handle execution errors
                fn = available_functions.get(
                    function_name, lambda: None
                )  # Ensure fn is always a callable
                logging.error(f"Error executing function '{function_name}': {e}")
                crewai_event_bus.emit(
                    self,
                    event=LLMCallFailedEvent(
                        error=f"Tool execution error: {e!s}",
                        from_task=from_task,
                        from_agent=from_agent,
                        call_id=get_current_call_id(),
                    ),
                )
                crewai_event_bus.emit(
                    self,
                    event=ToolUsageErrorEvent(
                        tool_name=function_name,
                        tool_args=function_args,
                        error=f"Tool execution error: {e!s}",
                        from_task=from_task,
                        from_agent=from_agent,
                    ),
                )
        return None

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
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
            from_task: Optional Task that invoked the LLM
            from_agent: Optional Agent that invoked the LLM
            response_model: Optional Model that contains a pydantic response model.

        Returns:
            Union[str, Any]: Either a text response from the LLM (str) or
                           the result of a tool function call (Any).

        Raises:
            TypeError: If messages format is invalid
            ValueError: If response format is not supported
            LLMContextLengthExceededError: If input exceeds model's context limit
        """
        with llm_call_context() as call_id:
            crewai_event_bus.emit(
                self,
                event=LLMCallStartedEvent(
                    messages=messages,
                    tools=tools,
                    callbacks=callbacks,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    model=self.model,
                    call_id=call_id,
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
                        msg_role: Literal["assistant"] = "assistant"
                        message["role"] = msg_role

            if not self._invoke_before_llm_call_hooks(messages, from_agent):
                raise ValueError("LLM call blocked by before_llm_call hook")

            # --- 5) Set up callbacks if provided
            with suppress_warnings():
                if callbacks and len(callbacks) > 0:
                    self.set_callbacks(callbacks)
                try:
                    # --- 6) Prepare parameters for the completion call
                    params = self._prepare_completion_params(messages, tools)
                    # --- 7) Make the completion call and handle response
                    if self.stream:
                        result = self._handle_streaming_response(
                            params=params,
                            callbacks=callbacks,
                            available_functions=available_functions,
                            from_task=from_task,
                            from_agent=from_agent,
                            response_model=response_model,
                        )
                    else:
                        result = self._handle_non_streaming_response(
                            params=params,
                            callbacks=callbacks,
                            available_functions=available_functions,
                            from_task=from_task,
                            from_agent=from_agent,
                            response_model=response_model,
                        )

                    if isinstance(result, str):
                        result = self._invoke_after_llm_call_hooks(
                            messages, result, from_agent
                        )

                    return result
                except LLMContextLengthExceededError:
                    # Re-raise LLMContextLengthExceededError as it should be handled
                    # by the CrewAgentExecutor._invoke_loop method, which can then decide
                    # whether to summarize the content or abort based on the respect_context_window flag
                    raise
                except Exception as e:
                    error_str = str(e)
                    unsupported_stop = "'stop'" in error_str and (
                        "Unsupported parameter" in error_str
                        or "does not support parameters" in error_str
                    )

                    if unsupported_stop:
                        if (
                            "additional_drop_params" in self.additional_params
                            and isinstance(
                                self.additional_params["additional_drop_params"], list
                            )
                        ):
                            self.additional_params["additional_drop_params"].append(
                                "stop"
                            )
                        else:
                            self.additional_params = {
                                "additional_drop_params": ["stop"]
                            }

                        logging.info("Retrying LLM call without the unsupported 'stop'")

                        return self.call(
                            messages,
                            tools=tools,
                            callbacks=callbacks,
                            available_functions=available_functions,
                            from_task=from_task,
                            from_agent=from_agent,
                            response_model=response_model,
                        )

                    crewai_event_bus.emit(
                        self,
                        event=LLMCallFailedEvent(
                            error=str(e),
                            from_task=from_task,
                            from_agent=from_agent,
                            call_id=get_current_call_id(),
                        ),
                    )
                    raise

    async def acall(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Async high-level LLM call method.

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
            from_task: Optional Task that invoked the LLM
            from_agent: Optional Agent that invoked the LLM
            response_model: Optional Model that contains a pydantic response model.

        Returns:
            Union[str, Any]: Either a text response from the LLM (str) or
                           the result of a tool function call (Any).

        Raises:
            TypeError: If messages format is invalid
            ValueError: If response format is not supported
            LLMContextLengthExceededError: If input exceeds model's context limit
        """
        with llm_call_context() as call_id:
            crewai_event_bus.emit(
                self,
                event=LLMCallStartedEvent(
                    messages=messages,
                    tools=tools,
                    callbacks=callbacks,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    model=self.model,
                    call_id=call_id,
                ),
            )

            self._validate_call_params()

            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]

            # Process file attachments asynchronously before preparing params
            messages = await self._aprocess_message_files(messages)

            if "o1" in self.model.lower():
                for message in messages:
                    if message.get("role") == "system":
                        msg_role: Literal["assistant"] = "assistant"
                        message["role"] = msg_role

            with suppress_warnings():
                if callbacks and len(callbacks) > 0:
                    self.set_callbacks(callbacks)
                try:
                    params = self._prepare_completion_params(
                        messages, tools, skip_file_processing=True
                    )

                    if self.stream:
                        return await self._ahandle_streaming_response(
                            params=params,
                            callbacks=callbacks,
                            available_functions=available_functions,
                            from_task=from_task,
                            from_agent=from_agent,
                            response_model=response_model,
                        )

                    return await self._ahandle_non_streaming_response(
                        params=params,
                        callbacks=callbacks,
                        available_functions=available_functions,
                        from_task=from_task,
                        from_agent=from_agent,
                        response_model=response_model,
                    )
                except LLMContextLengthExceededError:
                    raise
                except Exception as e:
                    error_str = str(e)
                    unsupported_stop = "'stop'" in error_str and (
                        "Unsupported parameter" in error_str
                        or "does not support parameters" in error_str
                    )

                    if unsupported_stop:
                        if (
                            "additional_drop_params" in self.additional_params
                            and isinstance(
                                self.additional_params["additional_drop_params"], list
                            )
                        ):
                            self.additional_params["additional_drop_params"].append(
                                "stop"
                            )
                        else:
                            self.additional_params = {
                                "additional_drop_params": ["stop"]
                            }

                        logging.info("Retrying LLM call without the unsupported 'stop'")

                        return await self.acall(
                            messages,
                            tools=tools,
                            callbacks=callbacks,
                            available_functions=available_functions,
                            from_task=from_task,
                            from_agent=from_agent,
                            response_model=response_model,
                        )

                    crewai_event_bus.emit(
                        self,
                        event=LLMCallFailedEvent(
                            error=str(e),
                            from_task=from_task,
                            from_agent=from_agent,
                            call_id=get_current_call_id(),
                        ),
                    )
                    raise

    @staticmethod
    def _usage_to_dict(usage: Any) -> dict[str, Any] | None:
        if usage is None:
            return None
        if isinstance(usage, dict):
            return usage
        if isinstance(usage, BaseModel):
            result: dict[str, Any] = usage.model_dump()
            return result
        if hasattr(usage, "__dict__"):
            return {k: v for k, v in vars(usage).items() if not k.startswith("_")}
        return None

    def _handle_emit_call_events(
        self,
        response: Any,
        call_type: LLMCallType,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
        messages: str | list[LLMMessage] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        """Handle the events for the LLM call.

        Args:
            response (str): The response from the LLM call.
            call_type (str): The type of call, either "tool_call" or "llm_call".
            from_task: Optional task object
            from_agent: Optional agent object
            messages: Optional messages object
            usage: Optional token usage data
        """
        crewai_event_bus.emit(
            self,
            event=LLMCallCompletedEvent(
                messages=messages,
                response=response,
                call_type=call_type,
                from_task=from_task,
                from_agent=from_agent,
                model=self.model,
                call_id=get_current_call_id(),
                usage=usage,
            ),
        )

    def _process_message_files(self, messages: list[LLMMessage]) -> list[LLMMessage]:
        """Process files attached to messages and format for provider.

        For each message with a `files` field, formats the files into
        provider-specific content blocks and updates the message content.

        Args:
            messages: List of messages that may contain file attachments.

        Returns:
            Messages with files formatted into content blocks.
        """
        if not HAS_CREWAI_FILES:
            return messages

        if not self.supports_multimodal():
            if any(msg.get("files") for msg in messages):
                raise ValueError(
                    f"Model '{self.model}' does not support multimodal input, "
                    "but files were provided via 'input_files'. "
                    "Use a vision-capable model or remove the file inputs."
                )
            return messages

        provider = self.provider or self.model

        for msg in messages:
            files = msg.get("files")
            if not files:
                continue

            content_blocks = format_multimodal_content(files, provider)
            if not content_blocks:
                msg.pop("files", None)
                continue

            existing_content = msg.get("content", "")
            if isinstance(existing_content, str):
                msg["content"] = [
                    self.format_text_content(existing_content),
                    *content_blocks,
                ]
            elif isinstance(existing_content, list):
                msg["content"] = [*existing_content, *content_blocks]

            msg.pop("files", None)

        return messages

    async def _aprocess_message_files(
        self, messages: list[LLMMessage]
    ) -> list[LLMMessage]:
        """Async process files attached to messages and format for provider.

        For each message with a `files` field, formats the files into
        provider-specific content blocks and updates the message content.

        Args:
            messages: List of messages that may contain file attachments.

        Returns:
            Messages with files formatted into content blocks.
        """
        if not HAS_CREWAI_FILES:
            return messages

        if not self.supports_multimodal():
            if any(msg.get("files") for msg in messages):
                raise ValueError(
                    f"Model '{self.model}' does not support multimodal input, "
                    "but files were provided via 'input_files'. "
                    "Use a vision-capable model or remove the file inputs."
                )
            return messages

        provider = self.provider or self.model

        for msg in messages:
            files = msg.get("files")
            if not files:
                continue

            content_blocks = await aformat_multimodal_content(files, provider)
            if not content_blocks:
                msg.pop("files", None)
                continue

            existing_content = msg.get("content", "")
            if isinstance(existing_content, str):
                msg["content"] = [
                    self.format_text_content(existing_content),
                    *content_blocks,
                ]
            elif isinstance(existing_content, list):
                msg["content"] = [*existing_content, *content_blocks]

            msg.pop("files", None)

        return messages

    def _format_messages_for_provider(
        self, messages: list[LLMMessage]
    ) -> list[dict[str, str]]:
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
                    formatted_messages.append(msg)  # type: ignore[arg-type]
            return formatted_messages  # type: ignore[return-value]

        # Handle Mistral models - they require the last message to have a role of 'user' or 'tool'
        if "mistral" in self.model.lower():
            # Check if the last message has a role of 'assistant'
            if messages and messages[-1]["role"] == "assistant":
                return [*messages, {"role": "user", "content": "Please continue."}]  # type: ignore[list-item]
            return messages  # type: ignore[return-value]

        # TODO: Remove this code after merging PR https://github.com/BerriAI/litellm/pull/10917
        # Ollama doesn't supports last message to be 'assistant'
        if (
            "ollama" in self.model.lower()
            and messages
            and messages[-1]["role"] == "assistant"
        ):
            return [*messages, {"role": "user", "content": ""}]  # type: ignore[list-item]

        # Handle Anthropic models
        if not self.is_anthropic:
            return messages  # type: ignore[return-value]

        # Anthropic requires messages to start with 'user' role
        if not messages or messages[0]["role"] == "system":
            # If first message is system or empty, add a placeholder user message
            return [{"role": "user", "content": "."}, *messages]  # type: ignore[list-item]

        return messages  # type: ignore[return-value]

    def _get_custom_llm_provider(self) -> str | None:
        """
        Derives the custom_llm_provider from the model string.
        - For example, if the model is "openrouter/deepseek/deepseek-chat", returns "openrouter".
        - If the model is "gemini/gemini-1.5-pro", returns "gemini".
        - If there is no '/', defaults to "openai".
        """
        if "/" in self.model:
            return self.model.partition("/")[0]
        return None

    def _validate_call_params(self) -> None:
        """
        Validate parameters before making a call. Currently this only checks if
        a response_format is provided and whether the model supports it.
        The custom_llm_provider is dynamically determined from the model:
          - E.g., "openrouter/deepseek/deepseek-chat" yields "openrouter"
          - "gemini/gemini-1.5-pro" yields "gemini"
          - If no slash is present, "openai" is assumed.

        Note: This validation only applies to the litellm fallback path.
        Native providers have their own validation.
        """
        if not LITELLM_AVAILABLE or supports_response_schema is None:
            # When litellm is not available, skip validation
            # (this path should only be reached for litellm fallback models)
            return

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
        """Check if the model supports function calling.

        Note: This method is only used by the litellm fallback path.
        Native providers override this method with their own implementation.
        """
        if not LITELLM_AVAILABLE:
            # When litellm is not available, assume function calling is supported
            # (all modern models support it)
            return True

        try:
            provider = self._get_custom_llm_provider()
            return litellm.utils.supports_function_calling(
                self.model, custom_llm_provider=provider
            )
        except Exception as e:
            logging.error(f"Failed to check function calling support: {e!s}")
            return True  # Default to True for modern models

    def supports_stop_words(self) -> bool:
        """Check if the model supports stop words.

        Note: This method is only used by the litellm fallback path.
        Native providers override this method with their own implementation.
        """
        model_lower = self.model.lower() if self.model else ""
        if "gpt-5" in model_lower:
            return False

        if not LITELLM_AVAILABLE or get_supported_openai_params is None:
            # When litellm is not available, assume stop words are supported
            return True

        try:
            params = get_supported_openai_params(model=self.model)
            return params is not None and "stop" in params
        except Exception as e:
            logging.error(f"Failed to get supported params: {e!s}")
            return True  # Default to True

    def get_context_window_size(self) -> int:
        """
        Returns the context window size, using 75% of the maximum to avoid
        cutting off messages mid-thread.

        Raises:
            ValueError: If a model's context window size is outside valid bounds (1024-2097152)
        """
        if self.context_window_size != 0:
            return self.context_window_size

        min_context = 1024
        max_context = 2097152  # Current max from gemini-1.5-pro

        # Validate all context window sizes
        for key, value in LLM_CONTEXT_WINDOW_SIZES.items():
            if value < min_context or value > max_context:
                raise ValueError(
                    f"Context window for {key} must be between {min_context} and {max_context}"
                )

        self.context_window_size = int(
            DEFAULT_CONTEXT_WINDOW_SIZE * CONTEXT_WINDOW_USAGE_RATIO
        )
        for key, value in LLM_CONTEXT_WINDOW_SIZES.items():
            if self.model.startswith(key):
                self.context_window_size = int(value * CONTEXT_WINDOW_USAGE_RATIO)
        return self.context_window_size

    @staticmethod
    def set_callbacks(callbacks: list[Any]) -> None:
        """
        Attempt to keep a single set of callbacks in litellm by removing old
        duplicates and adding new ones.

        Note: This only affects the litellm fallback path. Native providers
        don't use litellm callbacks - they emit events via base_llm.py.
        """
        if not LITELLM_AVAILABLE:
            # When litellm is not available, callbacks are still stored
            # but not registered with litellm globals
            return

        with suppress_warnings():
            callback_types = [type(callback) for callback in callbacks]
            for callback in litellm.success_callback[:]:
                if type(callback) in callback_types:
                    litellm.success_callback.remove(callback)

            for callback in litellm._async_success_callback[:]:
                if type(callback) in callback_types:
                    litellm._async_success_callback.remove(callback)

            litellm.callbacks = callbacks

    @staticmethod
    def set_env_callbacks() -> None:
        """Sets the success and failure callbacks for the LiteLLM library from environment variables.

        This method reads the `LITELLM_SUCCESS_CALLBACKS` and `LITELLM_FAILURE_CALLBACKS`
        environment variables, which should contain comma-separated lists of callback names.
        It then assigns these lists to `litellm.success_callback` and `litellm.failure_callback`,
        respectively.

        If the environment variables are not set or are empty, the corresponding callback lists
        will be set to empty lists.

        Note: This only affects the litellm fallback path. Native providers
        don't use litellm callbacks - they emit events via base_llm.py.

        Examples:
            LITELLM_SUCCESS_CALLBACKS="langfuse,langsmith"
            LITELLM_FAILURE_CALLBACKS="langfuse"

        This will set `litellm.success_callback` to ["langfuse", "langsmith"] and
        `litellm.failure_callback` to ["langfuse"].
        """
        if not LITELLM_AVAILABLE:
            # When litellm is not available, env callbacks have no effect
            return

        with suppress_warnings():
            success_callbacks_str = os.environ.get("LITELLM_SUCCESS_CALLBACKS", "")
            success_callbacks: list[str | Callable[..., Any]] = []
            if success_callbacks_str:
                success_callbacks = [
                    cb.strip() for cb in success_callbacks_str.split(",") if cb.strip()
                ]

            failure_callbacks_str = os.environ.get("LITELLM_FAILURE_CALLBACKS", "")
            if failure_callbacks_str:
                failure_callbacks: list[str | Callable[..., Any]] = [
                    cb.strip() for cb in failure_callbacks_str.split(",") if cb.strip()
                ]

                litellm.success_callback = success_callbacks  # type: ignore[assignment]
                litellm.failure_callback = failure_callbacks  # type: ignore[assignment]

    def __copy__(self) -> LLM:
        """Create a shallow copy of the LLM instance."""
        # Filter out parameters that are already explicitly passed to avoid conflicts
        filtered_params = {
            k: v
            for k, v in self.additional_params.items()
            if k
            not in [
                "model",
                "is_litellm",
                "temperature",
                "top_p",
                "n",
                "max_completion_tokens",
                "max_tokens",
                "presence_penalty",
                "frequency_penalty",
                "logit_bias",
                "response_format",
                "seed",
                "logprobs",
                "top_logprobs",
                "base_url",
                "api_base",
                "api_version",
                "api_key",
                "callbacks",
                "reasoning_effort",
                "stream",
                "stop",
                "prefer_upload",
            ]
        }

        # Create a new instance with the same parameters
        return LLM(
            model=self.model,
            is_litellm=self.is_litellm,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
            max_completion_tokens=self.max_completion_tokens,
            max_tokens=self.max_tokens,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            logit_bias=self.logit_bias,
            response_format=self.response_format,
            seed=self.seed,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
            base_url=self.base_url,
            api_base=self.api_base,
            api_version=self.api_version,
            api_key=self.api_key,
            callbacks=self.callbacks,
            reasoning_effort=self.reasoning_effort,
            stream=self.stream,
            stop=self.stop,
            prefer_upload=self.prefer_upload,
            **filtered_params,
        )

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> LLM:
        """Create a deep copy of the LLM instance."""
        import copy

        # Filter out parameters that are already explicitly passed to avoid conflicts
        filtered_params = {
            k: copy.deepcopy(v, memo)
            for k, v in self.additional_params.items()
            if k
            not in [
                "model",
                "is_litellm",
                "temperature",
                "top_p",
                "n",
                "max_completion_tokens",
                "max_tokens",
                "presence_penalty",
                "frequency_penalty",
                "logit_bias",
                "response_format",
                "seed",
                "logprobs",
                "top_logprobs",
                "base_url",
                "api_base",
                "api_version",
                "api_key",
                "callbacks",
                "reasoning_effort",
                "stream",
                "stop",
                "prefer_upload",
            ]
        }

        # Create a new instance with the same parameters
        return LLM(
            model=self.model,
            is_litellm=self.is_litellm,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
            max_completion_tokens=self.max_completion_tokens,
            max_tokens=self.max_tokens,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            logit_bias=(
                copy.deepcopy(self.logit_bias, memo) if self.logit_bias else None
            ),
            response_format=(
                copy.deepcopy(self.response_format, memo)
                if self.response_format
                else None
            ),
            seed=self.seed,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
            base_url=self.base_url,
            api_base=self.api_base,
            api_version=self.api_version,
            api_key=self.api_key,
            callbacks=copy.deepcopy(self.callbacks, memo) if self.callbacks else None,
            reasoning_effort=self.reasoning_effort,
            stream=self.stream,
            stop=copy.deepcopy(self.stop, memo) if self.stop else None,
            prefer_upload=self.prefer_upload,
            **filtered_params,
        )

    def supports_multimodal(self) -> bool:
        """Check if the model supports multimodal inputs.

        For litellm, check common vision-enabled model prefixes.

        Returns:
            True if the model likely supports images.
        """
        vision_prefixes = (
            # OpenAI — GPT-4 vision models
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4-vision",
            "gpt-4.1",
            # OpenAI — GPT-5 family (all variants support multimodal)
            "gpt-5",
            # OpenAI — o-series reasoning models with vision
            # o1, o3, o4, o4-mini support multimodal
            # o1-mini, o1-preview, o3-mini are text-only — handled via exclusion below
            "o1",
            "o3",
            "o4-mini",
            "o4",
            # Anthropic — Claude 3+ models support vision
            "claude-3",
            "claude-4",
            "claude-sonnet-4",
            "claude-opus-4",
            "claude-haiku-4",
            # Google — all Gemini models support multimodal
            "gemini",
            # xAI — Grok models support vision
            "grok",
            # Mistral — Pixtral vision model
            "pixtral",
            # Open-source vision models
            "llava",
            # Alibaba — Qwen vision-language models
            "qwen-vl",
            "qwen2-vl",
            "qwen3-vl",
        )
        # Text-only models that would otherwise match vision prefixes
        text_only_models = ("o3-mini", "o1-mini", "o1-preview")

        model_lower = self.model.lower()

        # Check exclusion first
        if any(
            model_lower.startswith(m) or f"/{m}" in model_lower
            for m in text_only_models
        ):
            return False

        return any(
            model_lower.startswith(p) or f"/{p}" in model_lower for p in vision_prefixes
        )
