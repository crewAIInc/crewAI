import json
import os
import subprocess
from typing import Any

import click


try:
    import weaviate
    from weaviate.classes.config import Configure, Vectorizers
    from weaviate.classes.init import Auth

    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    weaviate = Any  # type: ignore[assignment,misc]  # type placeholder
    Configure = Any  # type: ignore[assignment,misc]
    Vectorizers = Any  # type: ignore[assignment,misc]
    Auth = Any  # type: ignore[assignment,misc]

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


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
    headers: dict | None = None
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
        internal_docs = client.collections.get(self.collection_name)

        if not internal_docs:
            internal_docs = client.collections.create(
                name=self.collection_name,
                vectorizer_config=self.vectorizer,  # type: ignore
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
