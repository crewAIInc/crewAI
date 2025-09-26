"""Mistral embedding types."""

from typing import Any

from pydantic import Field

from crewai.rag.embeddings.providers.mistral.mistral_provider import MistralProvider


class MistralConfig:
    """Configuration for Mistral embeddings."""

    provider: str = Field(default="mistral", description="Provider name")
    api_key: str | None = Field(
        default=None, description="Mistral API key", validation_alias="MISTRAL_API_KEY"
    )
    model_name: str = Field(
        default="mistral-embed",
        description="Model name to use for embeddings",
        validation_alias="MISTRAL_MODEL_NAME",
    )
    base_url: str = Field(
        default="https://api.mistral.ai/v1",
        description="Base URL for API requests",
        validation_alias="MISTRAL_BASE_URL",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for API requests",
        validation_alias="MISTRAL_MAX_RETRIES",
    )
    timeout: int = Field(
        default=30,
        description="Timeout for API requests in seconds",
        validation_alias="MISTRAL_TIMEOUT",
    )

    def to_provider(self) -> MistralProvider:
        """Convert config to provider instance."""
        return MistralProvider(
            api_key=self.api_key,
            model_name=self.model_name,
            base_url=self.base_url,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )
