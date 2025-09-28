import os
from abc import ABC, abstractmethod
from typing import Any, cast

from crewai.rag.embeddings.factory import get_embedding_function
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Adapter(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def query(
        self,
        question: str,
        similarity_threshold: float | None = None,
        limit: int | None = None,
    ) -> str:
        """Query the knowledge base with a question and return the answer."""

    @abstractmethod
    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add content to the knowledge base."""


class RagTool(BaseTool):
    class _AdapterPlaceholder(Adapter):
        def query(
            self,
            question: str,
            similarity_threshold: float | None = None,
            limit: int | None = None,
        ) -> str:
            raise NotImplementedError

        def add(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError

    name: str = "Knowledge base"
    description: str = "A knowledge base that can be used to answer questions."
    summarize: bool = False
    similarity_threshold: float = 0.6
    limit: int = 5
    adapter: Adapter = Field(default_factory=_AdapterPlaceholder)
    config: Any | None = None

    @model_validator(mode="after")
    def _set_default_adapter(self):
        if isinstance(self.adapter, RagTool._AdapterPlaceholder):
            from crewai_tools.adapters.crewai_rag_adapter import CrewAIRagAdapter

            parsed_config = self._parse_config(self.config)

            self.adapter = CrewAIRagAdapter(
                collection_name="rag_tool_collection",
                summarize=self.summarize,
                similarity_threshold=self.similarity_threshold,
                limit=self.limit,
                config=parsed_config,
            )

        return self

    def _parse_config(self, config: Any) -> Any:
        """Parse complex config format to extract provider-specific config.

        Raises:
            ValueError: If the config format is invalid or uses unsupported providers.
        """
        if config is None:
            return None

        if isinstance(config, dict) and "provider" in config:
            return config

        if isinstance(config, dict):
            if "vectordb" in config:
                vectordb_config = config["vectordb"]
                if isinstance(vectordb_config, dict) and "provider" in vectordb_config:
                    provider = vectordb_config["provider"]
                    provider_config = vectordb_config.get("config", {})

                    supported_providers = ["chromadb", "qdrant"]
                    if provider not in supported_providers:
                        raise ValueError(
                            f"Unsupported vector database provider: '{provider}'. "
                            f"CrewAI RAG currently supports: {', '.join(supported_providers)}."
                        )

                    embedding_config = config.get("embedding_model")
                    embedding_function = None
                    if embedding_config and isinstance(embedding_config, dict):
                        embedding_function = self._create_embedding_function(
                            embedding_config, provider
                        )

                    return self._create_provider_config(
                        provider, provider_config, embedding_function
                    )
                else:
                    return None
            else:
                embedding_config = config.get("embedding_model")
                embedding_function = None
                if embedding_config and isinstance(embedding_config, dict):
                    embedding_function = self._create_embedding_function(
                        embedding_config, "chromadb"
                    )

                return self._create_provider_config("chromadb", {}, embedding_function)
        return config

    @staticmethod
    def _create_embedding_function(embedding_config: dict, provider: str) -> Any:
        """Create embedding function for the specified vector database provider."""
        embedding_provider = embedding_config.get("provider")
        embedding_model_config = embedding_config.get("config", {}).copy()

        if "model" in embedding_model_config:
            embedding_model_config["model_name"] = embedding_model_config.pop("model")

        factory_config = {"provider": embedding_provider, **embedding_model_config}

        if embedding_provider == "openai" and "api_key" not in factory_config:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                factory_config["api_key"] = api_key


        if provider == "chromadb":
            embedding_func = get_embedding_function(factory_config)
            return embedding_func

        elif provider == "qdrant":
            chromadb_func = get_embedding_function(factory_config)

            def qdrant_embed_fn(text: str) -> list[float]:
                """Embed text using ChromaDB function and convert to list of floats for Qdrant.

                Args:
                    text: The input text to embed.

                Returns:
                    A list of floats representing the embedding.
                """
                embeddings = chromadb_func([text])
                return embeddings[0] if embeddings and len(embeddings) > 0 else []

            return cast(Any, qdrant_embed_fn)

        return None

    @staticmethod
    def _create_provider_config(
        provider: str, provider_config: dict, embedding_function: Any
    ) -> Any:
        """Create proper provider config object."""
        if provider == "chromadb":
            from crewai.rag.chromadb.config import ChromaDBConfig

            config_kwargs = {}
            if embedding_function:
                config_kwargs["embedding_function"] = embedding_function

            config_kwargs.update(provider_config)

            return ChromaDBConfig(**config_kwargs)

        elif provider == "qdrant":
            from crewai.rag.qdrant.config import QdrantConfig

            config_kwargs = {}
            if embedding_function:
                config_kwargs["embedding_function"] = embedding_function

            config_kwargs.update(provider_config)

            return QdrantConfig(**config_kwargs)

        return None

    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.adapter.add(*args, **kwargs)

    def _run(
        self,
        query: str,
        similarity_threshold: float | None = None,
        limit: int | None = None,
    ) -> str:
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.similarity_threshold
        )
        result_limit = limit if limit is not None else self.limit
        return f"Relevant Content:\n{self.adapter.query(query, similarity_threshold=threshold, limit=result_limit)}"
