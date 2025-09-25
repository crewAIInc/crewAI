"""IBM Watson embeddings provider."""

from ibm_watsonx_ai import (  # type: ignore[import-not-found,import-untyped]
    APIClient,
    Credentials,
)
from pydantic import Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.providers.ibm.embedding_callable import (
    WatsonEmbeddingFunction,
)


class WatsonProvider(BaseEmbeddingsProvider[WatsonEmbeddingFunction]):
    """IBM Watson embeddings provider.

    Note: Requires custom implementation as Watson uses a different interface.
    """

    embedding_callable: type[WatsonEmbeddingFunction] = Field(
        default=WatsonEmbeddingFunction, description="Watson embedding function class"
    )
    model_id: str = Field(description="Watson model ID")
    params: dict[str, str | dict[str, str]] | None = Field(
        default=None, description="Additional parameters"
    )
    credentials: Credentials | None = Field(
        default=None, description="Watson credentials"
    )
    project_id: str | None = Field(default=None, description="Watson project ID")
    space_id: str | None = Field(default=None, description="Watson space ID")
    api_client: APIClient | None = Field(default=None, description="Watson API client")
    verify: bool | str | None = Field(default=None, description="SSL verification")
    persistent_connection: bool = Field(
        default=True, description="Use persistent connection"
    )
    batch_size: int = Field(default=100, description="Batch size for processing")
    concurrency_limit: int = Field(default=10, description="Concurrency limit")
    max_retries: int | None = Field(default=None, description="Maximum retries")
    delay_time: float | None = Field(
        default=None, description="Delay time between retries"
    )
    retry_status_codes: list[int] | None = Field(
        default=None, description="HTTP status codes to retry on"
    )
    url: str | None = Field(default=None, description="Watson API URL")
    api_key: str | None = Field(default=None, description="Watson API key")
    name: str | None = Field(default=None, description="Service name")
    iam_serviceid_crn: str | None = Field(
        default=None, description="IAM service ID CRN"
    )
    trusted_profile_id: str | None = Field(
        default=None, description="Trusted profile ID"
    )
    token: str | None = Field(default=None, description="Bearer token")
    projects_token: str | None = Field(default=None, description="Projects token")
    username: str | None = Field(default=None, description="Username")
    password: str | None = Field(default=None, description="Password")
    instance_id: str | None = Field(default=None, description="Service instance ID")
    version: str | None = Field(default=None, description="API version")
    bedrock_url: str | None = Field(default=None, description="Bedrock URL")
    platform_url: str | None = Field(default=None, description="Platform URL")
    proxies: dict | None = Field(default=None, description="Proxy configuration")
