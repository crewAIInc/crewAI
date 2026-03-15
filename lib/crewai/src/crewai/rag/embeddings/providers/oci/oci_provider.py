"""OCI embeddings provider."""

from typing import Any

from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.providers.oci.embedding_callable import OCIEmbeddingFunction


class OCIProvider(BaseEmbeddingsProvider[OCIEmbeddingFunction]):
    """OCI Generative AI embeddings provider."""

    embedding_callable: type[OCIEmbeddingFunction] = Field(
        default=OCIEmbeddingFunction,
        description="OCI embedding function class",
    )
    model_name: str = Field(
        default="cohere.embed-english-v3.0",
        description="Model name to use for embeddings",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OCI_MODEL_NAME",
            "OCI_EMBED_MODEL",
            "model",
            "model_name",
        ),
    )
    compartment_id: str = Field(
        description="OCI compartment ID",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OCI_COMPARTMENT_ID",
            "OCI_COMPARTMENT_ID",
            "compartment_id",
        ),
    )
    service_endpoint: str | None = Field(
        default=None,
        description="OCI Generative AI inference endpoint",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OCI_SERVICE_ENDPOINT",
            "OCI_SERVICE_ENDPOINT",
            "service_endpoint",
        ),
    )
    region: str | None = Field(
        default=None,
        description="OCI region used to derive the inference endpoint when service_endpoint is not provided",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OCI_REGION",
            "OCI_REGION",
            "region",
        ),
    )
    auth_type: str = Field(
        default="API_KEY",
        description="OCI SDK auth type",
        validation_alias=AliasChoices("EMBEDDINGS_OCI_AUTH_TYPE", "OCI_AUTH_TYPE"),
    )
    auth_profile: str = Field(
        default="DEFAULT",
        description="OCI config profile name",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OCI_AUTH_PROFILE",
            "OCI_AUTH_PROFILE",
        ),
    )
    auth_file_location: str = Field(
        default="~/.oci/config",
        description="OCI config file location",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OCI_AUTH_FILE_LOCATION",
            "OCI_AUTH_FILE_LOCATION",
        ),
    )
    truncate: str = Field(default="END", description="OCI embedding truncate policy")
    input_type: str | None = Field(
        default=None,
        description="Optional OCI embedding input type such as SEARCH_DOCUMENT or SEARCH_QUERY",
    )
    output_dimensions: int | None = Field(
        default=None,
        description="Optional output dimensions for compatible OCI embedding models",
    )
    batch_size: int = Field(default=96, description="OCI embedding batch size")
    timeout: tuple[int, int] = Field(
        default=(10, 120), description="OCI SDK connect/read timeout"
    )
    client: Any | None = Field(default=None, description="Injected OCI client")
