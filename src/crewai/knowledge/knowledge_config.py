from pydantic import BaseModel, Field


class KnowledgeConfig(BaseModel):
    results_limit: int = Field(default=3, description="The number of results to return")
    score_threshold: float = Field(
        default=0.35,
        description="The minimum score for a result to be considered relevant",
    )
