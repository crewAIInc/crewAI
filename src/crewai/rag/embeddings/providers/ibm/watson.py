"""IBM Watson embeddings provider."""

from typing import Any

from pydantic import AliasChoices, Field, model_validator
from typing_extensions import Self

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
    model_id: str = Field(
        description="Watson model ID",
        validation_alias=AliasChoices("WATSONX_MODEL_ID", "WATSON_MODEL_ID"),
    )
    params: dict[str, str | dict[str, str]] | None = Field(
        default=None, description="Additional parameters"
    )
    credentials: Any | None = Field(default=None, description="Watson credentials")
    project_id: str | None = Field(
        default=None,
        description="Watson project ID",
        validation_alias=AliasChoices("WATSONX_PROJECT_ID", "WATSON_PROJECT_ID"),
    )
    space_id: str | None = Field(
        default=None, description="Watson space ID", validation_alias="WATSON_SPACE_ID"
    )
    api_client: Any | None = Field(default=None, description="Watson API client")
    verify: bool | str | None = Field(
        default=None, description="SSL verification", validation_alias="WATSON_VERIFY"
    )
    persistent_connection: bool = Field(
        default=True,
        description="Use persistent connection",
        validation_alias="WATSON_PERSISTENT_CONNECTION",
    )
    batch_size: int = Field(
        default=100,
        description="Batch size for processing",
        validation_alias="WATSON_BATCH_SIZE",
    )
    concurrency_limit: int = Field(
        default=10,
        description="Concurrency limit",
        validation_alias="WATSON_CONCURRENCY_LIMIT",
    )
    max_retries: int | None = Field(
        default=None,
        description="Maximum retries",
        validation_alias="WATSON_MAX_RETRIES",
    )
    delay_time: float | None = Field(
        default=None,
        description="Delay time between retries",
        validation_alias="WATSON_DELAY_TIME",
    )
    retry_status_codes: list[int] | None = Field(
        default=None, description="HTTP status codes to retry on"
    )
    url: str = Field(
        description="Watson API URL",
        validation_alias=AliasChoices("WATSONX_URL", "WATSON_URL"),
    )
    api_key: str = Field(
        description="Watson API key",
        validation_alias=AliasChoices("WATSONX_APIKEY", "WATSON_API_KEY"),
    )
    name: str | None = Field(
        default=None, description="Service name", validation_alias="WATSON_NAME"
    )
    iam_serviceid_crn: str | None = Field(
        default=None,
        description="IAM service ID CRN",
        validation_alias="WATSON_IAM_SERVICEID_CRN",
    )
    trusted_profile_id: str | None = Field(
        default=None,
        description="Trusted profile ID",
        validation_alias="WATSON_TRUSTED_PROFILE_ID",
    )
    token: str | None = Field(
        default=None,
        description="Bearer token",
        validation_alias=AliasChoices("WATSONX_TOKEN", "WATSON_TOKEN"),
    )
    projects_token: str | None = Field(
        default=None,
        description="Projects token",
        validation_alias="WATSON_PROJECTS_TOKEN",
    )
    username: str | None = Field(
        default=None, description="Username", validation_alias="WATSON_USERNAME"
    )
    password: str | None = Field(
        default=None, description="Password", validation_alias="WATSON_PASSWORD"
    )
    instance_id: str | None = Field(
        default=None,
        description="Service instance ID",
        validation_alias="WATSON_INSTANCE_ID",
    )
    version: str | None = Field(
        default=None, description="API version", validation_alias="WATSON_VERSION"
    )
    bedrock_url: str | None = Field(
        default=None, description="Bedrock URL", validation_alias="WATSON_BEDROCK_URL"
    )
    platform_url: str | None = Field(
        default=None, description="Platform URL", validation_alias="WATSON_PLATFORM_URL"
    )
    proxies: dict | None = Field(default=None, description="Proxy configuration")

    @model_validator(mode="after")
    def validate_space_or_project(self) -> Self:
        """Validate that either space_id or project_id is provided."""
        if not self.space_id and not self.project_id:
            raise ValueError("One of 'space_id' or 'project_id' must be provided")
        return self
