from __future__ import annotations

import os
from typing import Any

from crewai.llms.providers.openai.completion import OpenAICompletion


AVIAN_API_BASE = "https://api.avian.io/v1"

# Context window sizes for Avian-hosted models
AVIAN_CONTEXT_WINDOWS: dict[str, int] = {
    "deepseek/deepseek-v3.2": 164000,
    "moonshotai/kimi-k2.5": 131000,
    "z-ai/glm-5": 131000,
    "minimax/minimax-m2.5": 1000000,
}

# Sensible default for unknown future models on Avian
_DEFAULT_AVIAN_CONTEXT_WINDOW = 131000


class AvianCompletion(OpenAICompletion):
    """Avian native completion implementation.

    Avian provides an OpenAI-compatible API for accessing a curated set of
    high-performance language models. This provider subclasses OpenAICompletion
    and configures it to use the Avian API endpoint by default.

    Authentication is via the AVIAN_API_KEY environment variable.

    Available models:
        - deepseek/deepseek-v3.2  (164K context, 65K output)
        - moonshotai/kimi-k2.5    (131K context, 8K output)
        - z-ai/glm-5              (131K context, 16K output)
        - minimax/minimax-m2.5    (1M context, 1M output)

    Usage:
        from crewai import LLM

        llm = LLM(model="avian/deepseek/deepseek-v3.2")
    """

    def __init__(
        self,
        model: str = "deepseek/deepseek-v3.2",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Avian completion client.

        Args:
            model: The model identifier (e.g. "deepseek/deepseek-v3.2").
            api_key: Avian API key. Defaults to AVIAN_API_KEY env var.
            base_url: API base URL. Defaults to https://api.avian.io/v1.
            **kwargs: Additional arguments passed to OpenAICompletion.
        """
        resolved_api_key = api_key or os.getenv("AVIAN_API_KEY")
        if resolved_api_key is None:
            raise ValueError(
                "AVIAN_API_KEY is required. Set it as an environment variable "
                "or pass api_key to the constructor."
            )

        resolved_base_url = base_url or os.getenv("AVIAN_API_BASE") or AVIAN_API_BASE

        # Ensure provider is set to "avian"
        kwargs.setdefault("provider", "avian")

        super().__init__(
            model=model,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            **kwargs,
        )

    def get_context_window_size(self) -> int:
        """Get the context window size for the model.

        The inherited OpenAICompletion implementation only recognizes GPT-
        prefixed models and returns a misleadingly small default for Avian
        models.  This override provides the correct context window sizes.
        """
        from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO

        # Find the best match for the model name
        for model_prefix, size in AVIAN_CONTEXT_WINDOWS.items():
            if self.model.startswith(model_prefix):
                return int(size * CONTEXT_WINDOW_USAGE_RATIO)

        # Default for unknown Avian models
        return int(_DEFAULT_AVIAN_CONTEXT_WINDOW * CONTEXT_WINDOW_USAGE_RATIO)
