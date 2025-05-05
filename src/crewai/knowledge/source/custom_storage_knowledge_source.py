from typing import Optional

from pydantic import Field

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage


class CustomStorageKnowledgeSource(BaseKnowledgeSource):
    """A knowledge source that uses a pre-existing storage with embeddings."""

    collection_name: Optional[str] = Field(default=None)

    def validate_content(self):
        """No content to validate as we're using pre-existing storage."""
        pass

    def add(self) -> None:
        """No need to add content as we're using pre-existing storage."""
        pass
