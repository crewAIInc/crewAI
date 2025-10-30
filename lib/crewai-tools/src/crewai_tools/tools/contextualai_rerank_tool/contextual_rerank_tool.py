from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class ContextualAIRerankSchema(BaseModel):
    """Schema for contextual rerank tool."""

    query: str = Field(..., description="The search query to rerank documents against")
    documents: list[str] = Field(..., description="List of document texts to rerank")
    instruction: str | None = Field(
        default=None, description="Optional instruction for reranking behavior"
    )
    metadata: list[str] | None = Field(
        default=None, description="Optional metadata for each document"
    )
    model: str = Field(
        default="ctxl-rerank-en-v1-instruct", description="Reranker model to use"
    )


class ContextualAIRerankTool(BaseTool):
    """Tool to rerank documents using Contextual AI's instruction-following reranker."""

    name: str = "Contextual AI Document Reranker"
    description: str = (
        "Rerank documents using Contextual AI's instruction-following reranker"
    )
    args_schema: type[BaseModel] = ContextualAIRerankSchema

    api_key: str
    package_dependencies: list[str] = Field(
        default_factory=lambda: ["contextual-client"]
    )

    def _run(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
        metadata: list[str] | None = None,
        model: str = "ctxl-rerank-en-v1-instruct",
    ) -> str:
        """Rerank documents using Contextual AI's instruction-following reranker."""
        try:
            import json

            import requests

            base_url = "https://api.contextual.ai/v1"
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
            }

            payload = {"query": query, "documents": documents, "model": model}

            if instruction:
                payload["instruction"] = instruction

            if metadata:
                if len(metadata) != len(documents):
                    raise ValueError(
                        "Metadata list must have the same length as documents list"
                    )
                payload["metadata"] = metadata

            rerank_url = f"{base_url}/rerank"
            result = requests.post(
                rerank_url, json=payload, headers=headers, timeout=30
            )

            if result.status_code != 200:
                raise RuntimeError(
                    f"Reranker API returned status {result.status_code}: {result.text}"
                )

            return json.dumps(result.json(), indent=2)

        except Exception as e:
            return f"Failed to rerank documents: {e!s}"
