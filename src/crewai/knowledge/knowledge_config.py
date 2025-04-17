from pydantic import BaseModel


class KnowledgeConfig(BaseModel):
    results_limit: int = 3
    score_threshold: float = 0.35
