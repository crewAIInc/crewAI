from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class KnowledgeConfig(BaseModel):
    """Configuration for knowledge querying behavior.

    Attributes:
        results_limit: Maximum number of results to return from a knowledge query.
        score_threshold: Minimum relevance score for results.
        metadata_filter: Metadata filter dict passed to the vector store query.
    """

    results_limit: int = Field(
        default=3,
        description="Maximum number of results to return from a knowledge query.",
    )
    score_threshold: float = Field(
        default=0.35,
        description="Minimum relevance score for results.",
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filter dict passed to the vector store query.",
    )
