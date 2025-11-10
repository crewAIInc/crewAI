"""Metaclass for LLM provider routing.

This metaclass enables automatic routing to native provider implementations
based on the model parameter at instantiation time.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic._internal._model_construction import ModelMetaclass


# Provider constants imported from crewai.llm.constants
SUPPORTED_NATIVE_PROVIDERS: list[str] = [
    "openai",
    "anthropic",
    "claude",
    "azure",
    "azure_openai",
    "google",
    "gemini",
    "bedrock",
    "aws",
]


class LLMMeta(ModelMetaclass):
    """Metaclass for LLM that handles provider routing.

    This metaclass intercepts LLM instantiation and routes to the appropriate
    native provider implementation based on the model parameter.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:  # noqa: N805
        """Route to appropriate provider implementation at instantiation time.

        Args:
            *args: Positional arguments (model should be first for LLM class)
            **kwargs: Keyword arguments including model, is_litellm, etc.

        Returns:
            Instance of the appropriate provider class or LLM class

        Raises:
            ValueError: If model is not a valid string
        """
        if cls.__name__ != "LLM":
            return super().__call__(*args, **kwargs)

        model = kwargs.get("model") or (args[0] if args else None)
        is_litellm = kwargs.get("is_litellm", False)

        if not model or not isinstance(model, str):
            raise ValueError("Model must be a non-empty string")

        if args and not kwargs.get("model"):
            kwargs["model"] = args[0]
            args = args[1:]
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
                kwargs_copy = {k: v for k, v in kwargs.items() if k not in ("provider", "model")}
                return native_class(
                    model=model_string, provider=provider, **kwargs_copy
                )
            except NotImplementedError:
                raise
            except Exception as e:
                raise ImportError(f"Error importing native provider: {e}") from e

        try:
            import litellm  # noqa: F401
        except ImportError:
            logging.error("LiteLLM is not available, falling back to LiteLLM")
            raise ImportError("Fallback to LiteLLM is not available") from None

        return super().__call__(model=model, is_litellm=True, **kwargs)

    @staticmethod
    def _validate_model_in_constants(model: str, provider: str) -> bool:
        """Validate if a model name exists in the provider's constants.

        Args:
            model: The model name to validate
            provider: The provider to check against (canonical name)

        Returns:
            True if the model exists in the provider's constants, False otherwise
        """
        from crewai.llm.constants import (
            ANTHROPIC_MODELS,
            BEDROCK_MODELS,
            GEMINI_MODELS,
            OPENAI_MODELS,
        )

        if provider == "openai":
            return model in OPENAI_MODELS

        if provider == "anthropic" or provider == "claude":
            return model in ANTHROPIC_MODELS

        if provider == "gemini":
            return model in GEMINI_MODELS

        if provider == "bedrock":
            return model in BEDROCK_MODELS

        if provider == "azure":
            # azure does not provide a list of available models
            return True

        return False

    @staticmethod
    def _infer_provider_from_model(model: str) -> str:
        """Infer the provider from the model name.

        Args:
            model: The model name without provider prefix

        Returns:
            The inferred provider name, defaults to "openai"
        """
        from crewai.llm.constants import (
            ANTHROPIC_MODELS,
            AZURE_MODELS,
            BEDROCK_MODELS,
            GEMINI_MODELS,
            OPENAI_MODELS,
        )

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

    @staticmethod
    def _get_native_provider(provider: str) -> type | None:
        """Get native provider class if available.

        Args:
            provider: The provider name

        Returns:
            The provider class or None if not available
        """
        if provider == "openai":
            from crewai.llm.providers.openai.completion import OpenAICompletion

            return OpenAICompletion

        if provider == "anthropic" or provider == "claude":
            from crewai.llm.providers.anthropic.completion import (
                AnthropicCompletion,
            )

            return AnthropicCompletion

        if provider == "azure" or provider == "azure_openai":
            from crewai.llm.providers.azure.completion import AzureCompletion

            return AzureCompletion

        if provider == "google" or provider == "gemini":
            from crewai.llm.providers.gemini.completion import GeminiCompletion

            return GeminiCompletion

        if provider == "bedrock":
            from crewai.llm.providers.bedrock.completion import BedrockCompletion

            return BedrockCompletion

        return None
