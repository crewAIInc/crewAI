from abc import ABC, abstractmethod
from typing import Any, Literal, cast

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.embeddings.factory import build_embedder
from crewai.rag.embeddings.types import ProviderSpec
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from crewai_tools.tools.rag.types import RagToolConfig, VectorDbConfig


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
    config: RagToolConfig = Field(
        default_factory=RagToolConfig,
        description="Configuration format accepted by RagTool.",
    )

    @model_validator(mode="after")
    def _ensure_adapter(self) -> Self:
        if isinstance(self.adapter, RagTool._AdapterPlaceholder):
            from crewai_tools.adapters.crewai_rag_adapter import CrewAIRagAdapter

            provider_cfg = self._parse_config(self.config)
            self.adapter = CrewAIRagAdapter(
                collection_name="rag_tool_collection",
                summarize=self.summarize,
                similarity_threshold=self.similarity_threshold,
                limit=self.limit,
                config=provider_cfg,
            )
        return self

    def _parse_config(self, config: RagToolConfig) -> Any:
        """
        Normalize the RagToolConfig into a provider-specific config object.
        Defaults to 'chromadb' with no extra provider config if none is supplied.
        """
        if not config:
            return self._create_provider_config("chromadb", {}, None)

        vectordb_cfg = cast(VectorDbConfig, config.get("vectordb", {}))
        provider: Literal["chromadb", "qdrant"] = vectordb_cfg.get(
            "provider", "chromadb"
        )
        provider_config: dict[str, Any] = vectordb_cfg.get("config", {})

        supported = ("chromadb", "qdrant")
        if provider not in supported:
            raise ValueError(
                f"Unsupported vector database provider: '{provider}'. "
                f"CrewAI RAG currently supports: {', '.join(supported)}."
            )

        embedding_spec: ProviderSpec | None = config.get("embedding_model")
        embedding_function = build_embedder(embedding_spec) if embedding_spec else None
        return self._create_provider_config(
            provider, provider_config, embedding_function
        )

    @staticmethod
    def _create_provider_config(
        provider: Literal["chromadb", "qdrant"],
        provider_config: dict[str, Any],
        embedding_function: EmbeddingFunction[Any] | None,
    ) -> Any:
        """Instantiate provider config with optional embedding_function injected."""
        if provider == "chromadb":
            from crewai.rag.chromadb.config import ChromaDBConfig

            kwargs = dict(provider_config)
            if embedding_function is not None:
                kwargs["embedding_function"] = embedding_function
            return ChromaDBConfig(**kwargs)

        if provider == "qdrant":
            from crewai.rag.qdrant.config import QdrantConfig

            kwargs = dict(provider_config)
            if embedding_function is not None:
                kwargs["embedding_function"] = embedding_function
            return QdrantConfig(**kwargs)

        raise ValueError(f"Unhandled provider: {provider}")

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
