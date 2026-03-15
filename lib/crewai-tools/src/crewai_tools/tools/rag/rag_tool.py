from abc import ABC, abstractmethod
from typing import Any, Literal, cast

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.embeddings.factory import build_embedder
from crewai.rag.embeddings.types import ProviderSpec
from crewai.tools import BaseTool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from typing_extensions import Self, Unpack

from crewai_tools.tools.rag.types import (
    AddDocumentParams,
    ContentItem,
    RagToolConfig,
    VectorDbConfig,
)


def _validate_embedding_config(
    value: dict[str, Any] | ProviderSpec,
) -> dict[str, Any] | ProviderSpec:
    """Validate embedding config and provide clearer error messages for union validation.

    This pre-validator catches Pydantic ValidationErrors from the ProviderSpec union
    and provides a cleaner, more focused error message that only shows the relevant
    provider's validation errors instead of all 18 union members.

    Args:
        value: The embedding configuration dictionary or validated ProviderSpec.

    Returns:
        A validated ProviderSpec instance, or the original value if already validated
        or missing required fields.

    Raises:
        ValueError: If the configuration is invalid for the specified provider.
    """
    if not isinstance(value, dict):
        return value

    provider = value.get("provider")
    if not provider:
        return value

    try:
        type_adapter: TypeAdapter[ProviderSpec] = TypeAdapter(ProviderSpec)
        return type_adapter.validate_python(value)
    except ValidationError as e:
        provider_key = f"{provider.lower()}providerspec"
        provider_errors = [
            err for err in e.errors() if provider_key in str(err.get("loc", "")).lower()
        ]

        if provider_errors:
            error_msgs = []
            for err in provider_errors:
                loc_parts = err["loc"]
                if str(loc_parts[0]).lower() == provider_key:
                    loc_parts = loc_parts[1:]
                loc = ".".join(str(x) for x in loc_parts)
                error_msgs.append(f"  - {loc}: {err['msg']}")

            raise ValueError(
                f"Invalid configuration for embedding provider '{provider}':\n"
                + "\n".join(error_msgs)
            ) from e

        raise


class Adapter(BaseModel, ABC):
    """Abstract base class for RAG adapters."""

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
        *args: ContentItem,
        **kwargs: Unpack[AddDocumentParams],
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

        def add(
            self,
            *args: ContentItem,
            **kwargs: Unpack[AddDocumentParams],
        ) -> None:
            raise NotImplementedError

    name: str = "Knowledge base"
    description: str = "A knowledge base that can be used to answer questions."
    summarize: bool = False
    similarity_threshold: float = 0.6
    limit: int = 5
    collection_name: str = "rag_tool_collection"
    adapter: Adapter = Field(default_factory=_AdapterPlaceholder)
    config: RagToolConfig = Field(
        default_factory=RagToolConfig,
        description="Configuration format accepted by RagTool.",
    )

    @field_validator("config", mode="before")
    @classmethod
    def _validate_config(cls, value: Any) -> Any:
        """Validate config with improved error messages for embedding providers."""
        if not isinstance(value, dict):
            return value

        embedding_model = value.get("embedding_model")
        if embedding_model:
            try:
                value["embedding_model"] = _validate_embedding_config(embedding_model)
            except ValueError:
                raise

        return value

    @model_validator(mode="after")
    def _ensure_adapter(self) -> Self:
        if isinstance(self.adapter, RagTool._AdapterPlaceholder):
            from crewai_tools.adapters.crewai_rag_adapter import CrewAIRagAdapter

            provider_cfg = self._parse_config(self.config)
            self.adapter = CrewAIRagAdapter(
                collection_name=self.collection_name,
                summarize=self.summarize,
                similarity_threshold=self.similarity_threshold,
                limit=self.limit,
                config=provider_cfg,
            )
        return self

    def _parse_config(self, config: RagToolConfig) -> Any:
        """Normalize the RagToolConfig into a provider-specific config object.

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
        if embedding_spec:
            embedding_spec = cast(
                ProviderSpec, _validate_embedding_config(embedding_spec)
            )

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
        *args: ContentItem,
        **kwargs: Unpack[AddDocumentParams],
    ) -> None:
        """Add content to the knowledge base.


        Args:
            *args: Content items to add (strings, paths, or document dicts)
            data_type: DataType enum or string (e.g., "file", "pdf_file", "text")
            path: Path to file or directory, alias to positional arg
            file_path: Alias for path
            metadata: Additional metadata to attach to documents
            url: URL to fetch content from
            website: Website URL to scrape
            github_url: GitHub repository URL
            youtube_url: YouTube video URL
            directory_path: Path to directory

        Examples:
            rag_tool.add("path/to/document.pdf", data_type=DataType.PDF_FILE)

            # Keyword argument (documented API)
            rag_tool.add(path="path/to/document.pdf", data_type="file")
            rag_tool.add(file_path="path/to/document.pdf", data_type="pdf_file")

            # Auto-detect type from extension
            rag_tool.add("path/to/document.pdf")  # auto-detects PDF
        """
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
