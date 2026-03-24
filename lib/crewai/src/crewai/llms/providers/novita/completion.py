from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from crewai.llms.providers.openai.completion import OpenAICompletion

if TYPE_CHECKING:
    import httpx

    from crewai.llms.hooks.base import BaseInterceptor

NOVITA_BASE_URL = "https://api.novita.ai/openai"
NOVITA_DEFAULT_MODEL = "moonshotai/kimi-k2.5"


class NovitaCompletion(OpenAICompletion):
    """Novita AI completion implementation.

    Uses the OpenAI-compatible endpoint at https://api.novita.ai/openai.
    Inherits all functionality from OpenAICompletion.
    """

    def __init__(
        self,
        model: str = NOVITA_DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        temperature: float | None = None,
        provider: str | None = None,
        interceptor: BaseInterceptor[httpx.Request, httpx.Response] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Novita AI completion client.

        Args:
            model: Model ID, defaults to moonshotai/kimi-k2.5.
            api_key: Novita API key. Falls back to NOVITA_API_KEY env var.
            base_url: Override base URL. Defaults to https://api.novita.ai/openai.
            **kwargs: Passed through to OpenAICompletion.
        """
        resolved_api_key = api_key or os.getenv("NOVITA_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Novita AI API key is required. "
                "Set the NOVITA_API_KEY environment variable or pass api_key."
            )

        super().__init__(
            model=model,
            api_key=resolved_api_key,
            base_url=base_url or NOVITA_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
            provider=provider or "novita",
            interceptor=interceptor,
            # Force completions API — Responses API is OpenAI-specific
            api="completions",
            **kwargs,
        )

    def _get_client_params(self) -> dict[str, Any]:
        """Get client parameters with Novita AI defaults."""
        if self.api_key is None:
            self.api_key = os.getenv("NOVITA_API_KEY")
            if self.api_key is None:
                raise ValueError(
                    "Novita AI API key is required. "
                    "Set the NOVITA_API_KEY environment variable or pass api_key."
                )

        base_params = {
            "api_key": self.api_key,
            "base_url": self.base_url or NOVITA_BASE_URL,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }

        client_params = {k: v for k, v in base_params.items() if v is not None}

        if self.client_params:
            client_params.update(self.client_params)

        return client_params
