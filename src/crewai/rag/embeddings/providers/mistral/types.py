"""Mistral embedding types."""

from typing import Any, TypedDict

from pydantic import BaseModel, Field


class MistralProviderSpec(TypedDict, total=False):
    """Mistral provider specification."""
    provider: str
    api_key: str | None
    model_name: str
    model: str  # Alias for model_name
    base_url: str
    max_retries: int
    timeout: int


class MistralConfig(BaseModel):
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
    model: str | None = Field(
        default=None,
        description="Model name (alias for model_name)",
        validation_alias="MISTRAL_MODEL",
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

    def to_provider(self):
        """Convert config to provider instance."""
        from crewai.rag.embeddings.providers.mistral.mistral_provider import MistralProvider
        
        return MistralProvider(
            api_key=self.api_key,
            model_name=self.model_name,
            model=self.model,
            base_url=self.base_url,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )
