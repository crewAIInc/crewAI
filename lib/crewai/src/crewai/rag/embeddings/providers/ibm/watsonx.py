"""IBM WatsonX embeddings provider."""

from typing import Any

from pydantic import AliasChoices, Field, model_validator
from typing_extensions import Self

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.providers.ibm.embedding_callable import (
    WatsonXEmbeddingFunction,
)


class WatsonXProvider(BaseEmbeddingsProvider[WatsonXEmbeddingFunction]):
    """IBM WatsonX embeddings provider.

    Note: Requires custom implementation as WatsonX uses a different interface.
    """

    embedding_callable: type[WatsonXEmbeddingFunction] = Field(
        default=WatsonXEmbeddingFunction, description="WatsonX embedding function class"
    )
    model_id: str = Field(
        description="WatsonX model ID",
        validation_alias=AliasChoices(
            "EMBEDDINGS_WATSONX_MODEL_ID", "WATSONX_MODEL_ID"
        ),
    )
    params: dict[str, str | dict[str, str]] | None = Field(
        default=None, description="Additional parameters"
    )
    credentials: Any | None = Field(default=None, description="WatsonX credentials")
    project_id: str | None = Field(
        default=None,
        description="WatsonX project ID",
        validation_alias=AliasChoices(
            "EMBEDDINGS_WATSONX_PROJECT_ID", "WATSONX_PROJECT_ID"
        ),
    )
    space_id: str | None = Field(
        default=None,
        description="WatsonX space ID",
        validation_alias=AliasChoices(
            "EMBEDDINGS_WATSONX_SPACE_ID", "WATSONX_SPACE_ID"
        ),
    )
    api_client: Any | None = Field(default=None, description="WatsonX API client")
    verify: bool | str | None = Field(
        default=None,
        description="SSL verification",
        validation_alias=AliasChoices("EMBEDDINGS_WATSONX_VERIFY", "WATSONX_VERIFY"),
    )
    persistent_connection: bool = Field(
        default=True,
        description="Use persistent connection",
        validation_alias=AliasChoices(
            "EMBEDDINGS_WATSONX_PERSISTENT_CONNECTION", "WATSONX_PERSISTENT_CONNECTION"
        ),
    )
    batch_size: int = Field(
        default=100,
        description="Batch size for processing",
        validation_alias=AliasChoices(
            "EMBEDDINGS_WATSONX_BATCH_SIZE", "WATSONX_BATCH_SIZE"
        ),
    )
    concurrency_limit: int = Field(
        default=10,
        description="Concurrency limit",
        validation_alias=AliasChoices(
            "EMBEDDINGS_WATSONX_CONCURRENCY_LIMIT", "WATSONX_CONCURRENCY_LIMIT"
        ),
    )
    max_retries: int | None = Field(
        default=None,
        description="Maximum retries",
        validation_alias=AliasChoices(
            "EMBEDDINGS_WATSONX_MAX_RETRIES", "WATSONX_MAX_RETRIES"
        ),
    )
    delay_time: float | None = Field(
        default=None,
        description="Delay time between retries",
        validation_alias=AliasChoices(
            "EMBEDDINGS_WATSONX_DELAY_TIME", "WATSONX_DELAY_TIME"
        ),
    )
    retry_status_codes: list[int] | None = Field(
        default=None, description="HTTP status codes to retry on"
    )
    url: str = Field(
        description="WatsonX API URL",
        validation_alias=AliasChoices("EMBEDDINGS_WATSONX_URL", "WATSONX_URL"),
    )
    api_key: str = Field(
        description="WatsonX API key",
        validation_alias=AliasChoices("EMBEDDINGS_WATSONX_API_KEY", "WATSONX_API_KEY"),
    )
    name: str | None = Field(
        default=None,
        description="Service name",
        validation_alias=AliasChoices("EMBEDDINGS_WATSONX_NAME", "WATSONX_NAME"),
    )
    iam_serviceid_crn: str | None = Field(
        default=None,
        description="IAM service ID CRN",
        validation_alias=AliasChoices(
            "EMBEDDINGS_WATSONX_IAM_SERVICEID_CRN", "WATSONX_IAM_SERVICEID_CRN"
        ),
    )
    trusted_profile_id: str | None = Field(
        default=None,
        description="Trusted profile ID",
        validation_alias=AliasChoices(
            "EMBEDDINGS_WATSONX_TRUSTED_PROFILE_ID", "WATSONX_TRUSTED_PROFILE_ID"
        ),
    )
    token: str | None = Field(
        default=None,
        description="Bearer token",
        validation_alias=AliasChoices("EMBEDDINGS_WATSONX_TOKEN", "WATSONX_TOKEN"),
    )
    projects_token: str | None = Field(
        default=None,
        description="Projects token",
        validation_alias=AliasChoices(
            "EMBEDDINGS_WATSONX_PROJECTS_TOKEN", "WATSONX_PROJECTS_TOKEN"
        ),
    )
    username: str | None = Field(
        default=None,
        description="Username",
        validation_alias=AliasChoices(
            "EMBEDDINGS_WATSONX_USERNAME", "WATSONX_USERNAME"
        ),
    )
    password: str | None = Field(
        default=None,
        description="Password",
        validation_alias=AliasChoices(
            "EMBEDDINGS_WATSONX_PASSWORD", "WATSONX_PASSWORD"
        ),
    )
    instance_id: str | None = Field(
        default=None,
        description="Service instance ID",
        validation_alias=AliasChoices(
            "EMBEDDINGS_WATSONX_INSTANCE_ID", "WATSONX_INSTANCE_ID"
        ),
    )
    version: str | None = Field(
        default=None,
        description="API version",
        validation_alias=AliasChoices("EMBEDDINGS_WATSONX_VERSION", "WATSONX_VERSION"),
    )
    bedrock_url: str | None = Field(
        default=None,
        description="Bedrock URL",
        validation_alias=AliasChoices(
            "EMBEDDINGS_WATSONX_BEDROCK_URL", "WATSONX_BEDROCK_URL"
        ),
    )
    platform_url: str | None = Field(
        default=None,
        description="Platform URL",
        validation_alias=AliasChoices(
            "EMBEDDINGS_WATSONX_PLATFORM_URL", "WATSONX_PLATFORM_URL"
        ),
    )
    proxies: dict[str, Any] | None = Field(
        default=None, description="Proxy configuration"
    )

    @model_validator(mode="after")
    def validate_space_or_project(self) -> Self:
        """Validate that either space_id or project_id is provided."""
        if not self.space_id and not self.project_id:
            raise ValueError("One of 'space_id' or 'project_id' must be provided")
        return self
