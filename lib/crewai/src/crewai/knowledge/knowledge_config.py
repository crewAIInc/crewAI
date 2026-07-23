from typing import Any

from pydantic import BaseModel, Field


class KnowledgeConfig(BaseModel):
    """Configuration for knowledge retrieval.

    Args:
        results_limit (int): The number of relevant documents to return. Must be
            at least 1.
        score_threshold (float): The minimum score for a document to be
            considered relevant. Must be greater than 0 and less than or equal
            to 1.
        metadata_filter (dict[str, Any] | None): Optional metadata filter
            forwarded to the underlying knowledge storage so retrieval can be
            narrowed to documents whose stored metadata matches these keys and
            values.
    """

    results_limit: int = Field(
        default=5, ge=1, description="The number of results to return"
    )
    score_threshold: float = Field(
        default=0.6,
        gt=0,
        le=1,
        description="The minimum score for a result to be considered relevant",
    )
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional metadata filter passed to knowledge storage to restrict "
            "retrieval to documents whose metadata matches these key/value pairs."
        ),
    )
