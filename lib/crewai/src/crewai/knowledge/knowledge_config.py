from typing import Any

from pydantic import BaseModel, Field


class KnowledgeConfig(BaseModel):
    """Configuration for knowledge retrieval.

    Args:
        results_limit (int): The number of relevant documents to return.
        score_threshold (float): The minimum score for a document to be considered relevant.
        metadata_filter (dict[str, Any] | None): Optional metadata filter for knowledge search.
    """

    results_limit: int = Field(default=5, description="The number of results to return")
    score_threshold: float = Field(
        default=0.6,
        description="The minimum score for a result to be considered relevant",
    )
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata filter to apply when searching the knowledge base",
    )
