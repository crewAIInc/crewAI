from pydantic import BaseModel


class KnowledgeConfig(BaseModel):
    limit: int = 3
    score_threshold: float = 0.35
