from __future__ import annotations

import os
from typing import Any, cast

from pydantic import BaseModel, Field

from crewai_tools.tools.rag.rag_tool import RagTool


class FixedOCIKnowledgeBaseToolSchema(BaseModel):
    """Input for OCIKnowledgeBaseTool with a preconfigured source."""

    query: str = Field(
        ..., description="Mandatory query you want to use to search the knowledge base"
    )


class OCIKnowledgeBaseToolSchema(FixedOCIKnowledgeBaseToolSchema):
    """Input for OCIKnowledgeBaseTool."""

    knowledge_source: str = Field(
        ...,
        description=(
            "File path, directory path, URL, or other CrewAI-supported source "
            "to add to the OCI-backed knowledge base before querying"
        ),
    )


class OCIKnowledgeBaseTool(RagTool):
    """RAG tool preconfigured to use OCI embeddings as the backing embedder."""
    name: str = "OCI Knowledge Base Tool"
    description: str = (
        "A CrewAI-managed knowledge base tool powered by OCI embeddings."
    )
    args_schema: type[BaseModel] = OCIKnowledgeBaseToolSchema

    def __init__(
        self,
        knowledge_source: str | None = None,
        *,
        model_name: str | None = None,
        compartment_id: str | None = None,
        service_endpoint: str | None = None,
        region: str | None = None,
        auth_type: str = "API_KEY",
        auth_profile: str | None = None,
        auth_file_location: str | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Keep the OCI embedder config serializable so the underlying RagTool can
        # build or rebuild the embedder through CrewAI's standard provider factory.
        oci_embedding_config: dict[str, str] = {
            "model_name": cast(
                str,
                model_name or os.getenv("OCI_EMBED_MODEL", "cohere.embed-english-v3.0"),
            ),
            "compartment_id": cast(
                str,
                compartment_id or os.getenv("OCI_COMPARTMENT_ID", ""),
            ),
            "region": cast(str, region or os.getenv("OCI_REGION", "eu-frankfurt-1")),
            "auth_type": auth_type,
            "auth_profile": cast(
                str,
                auth_profile or os.getenv("OCI_AUTH_PROFILE", "DEFAULT"),
            ),
            "auth_file_location": cast(
                str,
                auth_file_location or os.getenv("OCI_AUTH_FILE_LOCATION", "~/.oci/config"),
            ),
        }
        if service_endpoint or os.getenv("OCI_SERVICE_ENDPOINT"):
            oci_embedding_config["service_endpoint"] = (
                service_endpoint or os.getenv("OCI_SERVICE_ENDPOINT") or ""
            )

        merged_config = dict(config or {})
        merged_config.setdefault(
            "embedding_model",
            {
                "provider": "oci",
                "config": oci_embedding_config,
            },
        )

        super().__init__(config=merged_config, **kwargs)

        if knowledge_source is not None:
            self.add(knowledge_source)
            self.description = (
                "A CrewAI-managed knowledge base tool powered by OCI embeddings "
                f"and preloaded with {knowledge_source}."
            )
            self.args_schema = FixedOCIKnowledgeBaseToolSchema
            self._generate_description()

    def _run(  # type: ignore[override]
        self,
        query: str,
        knowledge_source: str | None = None,
        similarity_threshold: float | None = None,
        limit: int | None = None,
    ) -> str:
        """Optionally add a source, then delegate retrieval to the base RagTool."""
        if knowledge_source is not None:
            self.add(knowledge_source)
        return super()._run(
            query=query, similarity_threshold=similarity_threshold, limit=limit
        )
