import importlib.metadata
import json
import logging
import os
import subprocess
from typing import Any, Literal

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

    return Configure.Vectors.text2vec_openai(
        model="text-embedding-3-small",
    )


def _vector_config_kwargs(vectorizer: Any) -> dict[str, Any]:
    """Map `vectorizer` to the matching `collections.create` argument.

    `Configure.Vectors.*` (which carries `quantizer=` and `vector_index_config=`) goes to
    `vector_config`; the legacy `Configure.Vectorizer.*` still goes to `vectorizer_config`
    so existing tool configurations keep working.
    """
    from weaviate.collections.classes.config_vectorizers import (
        _VectorizerConfigCreate,
    )

    if isinstance(vectorizer, _VectorizerConfigCreate):
        return {"vectorizer_config": vectorizer}
    return {"vector_config": vectorizer}


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
    search_type: Literal["hybrid", "near_text", "bm25"] = Field(
        default="hybrid",
        description="Search to run: `hybrid` (vector + keyword, weighted by `alpha`), "
        "`near_text` (vector only) or `bm25` (keyword only)",
    )
    target_vector: str | list[str] | None = Field(
        default=None,
        description="Named vector(s) to search, for collections with more than one vector",
    )
    return_properties: list[str] | None = Field(
        default=None,
        description="Properties to return for each result; all properties when unset",
    )
    filters: Any = Field(
        default=None,
        description="A `weaviate.classes.query.Filter` applied to every search",
    )
    tenant: str | None = Field(
        default=None,
        description="Tenant to search in a multi-tenant collection. Set on the tool, never "
        "by the agent, so an agent cannot reach another tenant's data",
    )
    create_collection_if_missing: bool = Field(
        default=False,
        description="Create `collection_name` with `vectorizer` and `generative_model` "
        "when it does not exist; multi-tenant when `tenant` is set",
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="OPENAI_API_KEY",
                description="OpenAI API key, forwarded to Weaviate for OpenAI vectorization",
                required=False,
            ),
        ]
    )
    weaviate_cluster_url: str | None = Field(
        default=None,
        description="The URL of a Weaviate Cloud cluster. When unset, the tool connects to "
        "`weaviate_host` instead",
    )
    weaviate_api_key: str | None = Field(
        default=None,
        description="The API key for the Weaviate cluster",
    )
    weaviate_host: str = Field(
        default="localhost",
        description="Host of a self-hosted Weaviate, used when `weaviate_cluster_url` is unset",
    )
    weaviate_http_port: int = Field(default=8080)
    weaviate_grpc_port: int = Field(default=50051)
    weaviate_secure: bool = Field(
        default=False,
        description="Use HTTPS and TLS gRPC for a self-hosted Weaviate",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if WEAVIATE_AVAILABLE:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key:
                self.headers = {
                    "X-OpenAI-Api-Key": openai_api_key,
                    **(self.headers or {}),
                }
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

        client = self._connect()
        try:
            _register_integration(client)
            collection, created = self._collection(client)
            if created:
                # Nothing to find yet, and Weaviate rejects searching a collection that
                # has no properties.
                return ""
            response = self._search(collection, query)
        finally:
            client.close()

        json_response = ""
        for obj in response.objects:
            json_response += json.dumps(obj.properties, indent=2, default=str)
        return json_response

    def _connect(self) -> Any:
        auth = Auth.api_key(self.weaviate_api_key) if self.weaviate_api_key else None
        if self.weaviate_cluster_url:
            if auth is None:
                raise ValueError(
                    "weaviate_api_key is required with weaviate_cluster_url"
                )
            return weaviate.connect_to_weaviate_cloud(
                cluster_url=self.weaviate_cluster_url,
                auth_credentials=auth,
                headers=self.headers,
            )
        return weaviate.connect_to_custom(
            http_host=self.weaviate_host,
            http_port=self.weaviate_http_port,
            http_secure=self.weaviate_secure,
            grpc_host=self.weaviate_host,
            grpc_port=self.weaviate_grpc_port,
            grpc_secure=self.weaviate_secure,
            headers=self.headers,
            auth_credentials=auth,
        )

    def _collection(self, client: Any) -> tuple[Any, bool]:
        """The collection to search, and whether this call created it."""
        created = self.create_collection_if_missing and not client.collections.exists(
            self.collection_name
        )
        if created:
            client.collections.create(
                name=self.collection_name,
                generative_config=self.generative_model,
                multi_tenancy_config=Configure.multi_tenancy(
                    enabled=True, auto_tenant_creation=True
                )
                if self.tenant
                else None,
                **_vector_config_kwargs(self.vectorizer),
            )
        collection = client.collections.get(self.collection_name)
        if not self.tenant:
            return collection, created
        if created:
            # `auto_tenant_creation` only applies on insert; searching a tenant that does
            # not exist yet fails, so the new collection gets it up front.
            collection.tenants.create(self.tenant)
        return collection.with_tenant(self.tenant), created

    def _search(self, collection: Any, query: str) -> Any:
        common: dict[str, Any] = {
            "limit": self.limit,
            "filters": self.filters,
            "return_properties": self.return_properties,
        }
        if self.search_type == "bm25":
            return collection.query.bm25(query=query, **common)
        if self.search_type == "near_text":
            return collection.query.near_text(
                query=query, target_vector=self.target_vector, **common
            )
        return collection.query.hybrid(
            query=query, alpha=self.alpha, target_vector=self.target_vector, **common
        )
