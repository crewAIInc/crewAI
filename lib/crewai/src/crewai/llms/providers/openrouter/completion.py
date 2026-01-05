from __future__ import annotations

import os
from typing import Any

from crewai.llms.providers.openai.completion import OpenAICompletion


class OpenRouterCompletion(OpenAICompletion):
    """OpenRouter provider using OpenAI-compatible API.

    OpenRouter provides access to multiple LLM providers through a unified API.
    This class extends OpenAICompletion to use OpenRouter's endpoint while
    supporting OpenRouter-specific headers for attribution and ranking.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        site_url: str | None = None,
        site_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenRouter client.

        Args:
            model: The model identifier (e.g., "anthropic/claude-3.5-sonnet")
            api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
            site_url: Optional URL for OpenRouter rankings attribution.
                     Falls back to OPENROUTER_SITE_URL env var.
            site_name: Optional site name for OpenRouter rankings attribution.
                      Falls back to OPENROUTER_SITE_NAME env var.
            **kwargs: Additional arguments passed to OpenAICompletion.
        """
        # Remove base_url if passed (we override it)
        kwargs.pop("base_url", None)

        # Store OpenRouter-specific config before calling super().__init__
        self.site_url = site_url or os.getenv("OPENROUTER_SITE_URL", "")
        self.site_name = site_name or os.getenv("OPENROUTER_SITE_NAME", "")

        # Build default headers with OpenRouter-specific headers
        default_headers = kwargs.pop("default_headers", None) or {}
        if self.site_url:
            default_headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            default_headers["X-Title"] = self.site_name

        super().__init__(
            model=model,
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            default_headers=default_headers if default_headers else None,
            **kwargs,
        )

    def _get_client_params(self) -> dict[str, Any]:
        """Get OpenRouter client parameters.

        Overrides parent to use OPENROUTER_API_KEY instead of OPENAI_API_KEY.
        """
        if self.api_key is None:
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if self.api_key is None:
                raise ValueError("OPENROUTER_API_KEY is required")

        base_params = {
            "api_key": self.api_key,
            "organization": self.organization,
            "project": self.project,
            "base_url": self.base_url or "https://openrouter.ai/api/v1",
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }

        client_params = {k: v for k, v in base_params.items() if v is not None}

        if self.client_params:
            client_params.update(self.client_params)

        return client_params

    def get_context_window_size(self) -> int:
        """Get the context window size for the model.

        OpenRouter supports many models with varying context windows.
        Returns a conservative default since model capabilities vary.
        """
        from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO

        # Default context window for OpenRouter models
        # Users should check OpenRouter docs for specific model limits
        return int(128000 * CONTEXT_WINDOW_USAGE_RATIO)
