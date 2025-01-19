import json
import os
from typing import Any, Optional, Type

try:
    import weaviate
    from weaviate.classes.config import Configure, Vectorizers
    from weaviate.classes.init import Auth

    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    weaviate = Any  # type placeholder
    Configure = Any
    Vectorizers = Any
    Auth = Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class WeaviateToolSchema(BaseModel):
    """Input for WeaviateTool."""

    query: str = Field(
        ...,
        description="The query to search retrieve relevant information from the Weaviate database. Pass only the query, not the question.",
    )


class WeaviateVectorSearchTool(BaseTool):
    """Tool to search the Weaviate database"""

    name: str = "WeaviateVectorSearchTool"
    description: str = "A tool to search the Weaviate database for relevant information on internal documents."
    args_schema: Type[BaseModel] = WeaviateToolSchema
    query: Optional[str] = None
    vectorizer: Optional[Vectorizers] = None
    generative_model: Optional[str] = None
    collection_name: Optional[str] = None
    limit: Optional[int] = Field(default=3)
    headers: Optional[dict] = None
    weaviate_cluster_url: str = Field(
        ...,
        description="The URL of the Weaviate cluster",
    )
    weaviate_api_key: str = Field(
        ...,
        description="The API key for the Weaviate cluster",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if WEAVIATE_AVAILABLE:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required for WeaviateVectorSearchTool and it is mandatory to use the tool."
                )
            self.headers = {"X-OpenAI-Api-Key": openai_api_key}
            self.vectorizer = self.vectorizer or Configure.Vectorizer.text2vec_openai(
                model="nomic-embed-text",
            )
            self.generative_model = (
                self.generative_model
                or Configure.Generative.openai(
                    model="gpt-4o",
                )
            )

    def _run(self, query: str) -> str:
        if not WEAVIATE_AVAILABLE:
            raise ImportError(
                "The 'weaviate-client' package is required to use the WeaviateVectorSearchTool. "
                "Please install it with: uv add weaviate-client"
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
                vectorizer_config=self.vectorizer,
                generative_config=self.generative_model,
            )

        response = internal_docs.query.near_text(
            query=query,
            limit=self.limit,
        )
        json_response = ""
        for obj in response.objects:
            json_response += json.dumps(obj.properties, indent=2)

        client.close()
        return json_response
