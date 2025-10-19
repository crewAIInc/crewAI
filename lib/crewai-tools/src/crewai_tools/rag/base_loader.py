from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.rag.misc import compute_sha256
from crewai_tools.rag.source_content import SourceContent


class LoaderResult(BaseModel):
    content: str = Field(description="The text content of the source")
    source: str = Field(description="The source of the content", default="unknown")
    metadata: dict[str, Any] = Field(
        description="The metadata of the source", default_factory=dict
    )
    doc_id: str = Field(description="The id of the document")


class BaseLoader(ABC):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def load(self, content: SourceContent, **kwargs) -> LoaderResult: ...

    @staticmethod
    def generate_doc_id(
        source_ref: str | None = None, content: str | None = None
    ) -> str:
        """Generate a unique document id based on the source reference and content.
        If the source reference is not provided, the content is used as the source reference.
        If the content is not provided, the source reference is used as the content.
        If both are provided, the source reference is used as the content.

        Both are optional because the TEXT content type does not have a source reference. In this case, the content is used as the source reference.
        """
        source_ref = source_ref or ""
        content = content or ""

        return compute_sha256(source_ref + content)
