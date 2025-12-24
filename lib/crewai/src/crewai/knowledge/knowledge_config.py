from pydantic import BaseModel, Field


class KnowledgeConfig(BaseModel):
    """Configuratie voor kennis ophalen.

    Args:
        results_limit (int): Het aantal relevante documenten om te retourneren.
        score_threshold (float): De minimale score voor een document om als relevant te worden beschouwd.
    """

    results_limit: int = Field(default=5, description="Het aantal resultaten om te retourneren")
    score_threshold: float = Field(
        default=0.6,
        description="De minimale score voor een resultaat om als relevant te worden beschouwd",
    )
