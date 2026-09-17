import importlib.metadata
import json
import logging
import os
import subprocess
from typing import Any

import click


try:
    import weaviate
    from weaviate.classes.config import Configure, Vectorizers  # noqa: F401
    from weaviate.classes.init import Auth

    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    weaviate = Any  # type: ignore[assignment]
    Configure = Any  # type: ignore[assignment,misc]
    Auth = Any  # type: ignore[assignment,misc]

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

# Reported to Weaviate as the `X-Weaviate-Client-Integration` header so that server-side
# telemetry can attribute traffic to CrewAI, the same way `MongoDBVectorSearchTool` tags
# pymongo with `DriverInfo(name="CrewAI", ...)`. The version comes from `crewai-tools`,
# the distribution that ships this tool.
_INTEGRATION_NAME = "crewai-tools"


def _integration_version() -> str:
    """Version reported with `_INTEGRATION_NAME`; "unknown" when it cannot be resolved.

    `crewai_tools.__version__` is unusable here because `crewai_tools/__init__.py` imports
    this module, so the distribution metadata is read instead.
    """
    try:
        return importlib.metadata.version("crewai-tools").strip() or "unknown"
    except Exception:  # telemetry must never break the tool
        return "unknown"


def _register_integration(client: Any) -> None:
    """Best-effort: tag `client` with the `X-Weaviate-Client-Integration` header.

    `client.integrations.configure` applies the header to both the HTTP and the gRPC
    transport, which a plain `headers=` argument does not -- searches travel over gRPC.
    `_IntegrationConfig` is private API, so both the import and the subclass live inside the
    `try`: on a `weaviate-client` that lacks it, registration is skipped rather than breaking
    the tool. Weaviate parses the value as `{name}/{version}` and ignores it entirely before
    server version 1.38.0.
    """
    try:
        from pydantic import Field as PydanticField
        from weaviate.connect.integrations import _IntegrationConfig

        value = f"{_INTEGRATION_NAME}/{_integration_version()}"

        class _CrewAIIntegration(_IntegrationConfig):
            integration: str = PydanticField(
                default=value,
                serialization_alias="X-Weaviate-Client-Integration",
            )

        client.integrations.configure(_CrewAIIntegration())
    except Exception:  # telemetry must never break the tool
        logger.debug(
            "Could not register the CrewAI Weaviate integration header", exc_info=True
        )


class WeaviateToolSchema(BaseModel):
    """Input for WeaviateTool."""

    query: str = Field(
        ...,
        description="The query to search retrieve relevant information from the Weaviate database. Pass only the query, not the question.",
    )


def _set_generative_model() -> Any:
    """Set the generative model based on the provided model name."""
    from weaviate.classes.config import Configure

    return Configure.Generative.openai(
        model="gpt-4o",
    )


def _set_vectorizer() -> Any:
    """Set the vectorizer based on the provided model name."""
    from weaviate.classes.config import Configure

    return Configure.Vectorizer.text2vec_openai(
        model="nomic-embed-text",
    )


class WeaviateVectorSearchTool(BaseTool):
    """Tool to search the Weaviate database."""

    package_dependencies: list[str] = Field(default_factory=lambda: ["weaviate-client"])
    name: str = "WeaviateVectorSearchTool"
    description: str = "A tool to search the Weaviate database for relevant information on internal documents."
    args_schema: type[BaseModel] = WeaviateToolSchema
    query: str | None = None
    vectorizer: Any = Field(default_factory=_set_vectorizer)
    generative_model: Any = Field(default_factory=_set_generative_model)
    collection_name: str = Field(
        description="The name of the Weaviate collection to search",
    )
    limit: int | None = Field(default=3)
    headers: dict[str, str] | None = None
    alpha: float = Field(default=0.75)
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="OPENAI_API_KEY",
                description="OpenAI API key for embedding generation and retrieval",
                required=True,
            ),
        ]
    )
    weaviate_cluster_url: str = Field(
        ...,
        description="The URL of the Weaviate cluster",
    )
    weaviate_api_key: str = Field(
        ...,
        description="The API key for the Weaviate cluster",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if WEAVIATE_AVAILABLE:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required for WeaviateVectorSearchTool and it is mandatory to use the tool."
                )
            self.headers = {"X-OpenAI-Api-Key": openai_api_key}
        else:
            if click.confirm(
                "You are missing the 'weaviate-client' package. Would you like to install it?"
            ):
                subprocess.run(["uv", "pip", "install", "weaviate-client"], check=True)  # noqa: S607

            else:
                raise ImportError(
                    "You are missing the 'weaviate-client' package. Would you like to install it?"
                )

    def _run(self, query: str) -> str:
        if not WEAVIATE_AVAILABLE:
            raise ImportError(
                "You are missing the 'weaviate-client' package. Would you like to install it?"
            )

        if not self.weaviate_cluster_url or not self.weaviate_api_key:
            raise ValueError("WEAVIATE_URL or WEAVIATE_API_KEY is not set")

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.weaviate_cluster_url,
            auth_credentials=Auth.api_key(self.weaviate_api_key),
            headers=self.headers,
        )
        _register_integration(client)
        internal_docs = client.collections.get(self.collection_name)

        if not internal_docs:
            internal_docs = client.collections.create(
                name=self.collection_name,
                vectorizer_config=self.vectorizer,
                generative_config=self.generative_model,
            )

        response = internal_docs.query.hybrid(
            query=query, limit=self.limit, alpha=self.alpha
        )
        json_response = ""
        for obj in response.objects:
            json_response += json.dumps(obj.properties, indent=2)

        client.close()
        return json_response
