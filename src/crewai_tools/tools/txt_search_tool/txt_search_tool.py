from typing import Any, Optional, Type

from embedchain.models.data_type import DataType
from pydantic import BaseModel, Field

from ..rag.rag_tool import RagTool


class FixedTXTSearchToolSchema(BaseModel):
    """Input for TXTSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the txt's content",
    )


class TXTSearchToolSchema(FixedTXTSearchToolSchema):
    """Input for TXTSearchTool."""

    txt: str = Field(..., description="Mandatory txt path you want to search")


class TXTSearchTool(RagTool):
    name: str = "Search a txt's content"
    description: str = (
        "A tool that can be used to semantic search a query from a txt's content."
    )
    args_schema: Type[BaseModel] = TXTSearchToolSchema

    def __init__(self, txt: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if txt is not None:
            self.add(txt)
            self.description = f"A tool that can be used to semantic search a query the {txt} txt's content."
            self.args_schema = FixedTXTSearchToolSchema
            self._generate_description()

    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        kwargs["data_type"] = DataType.TEXT_FILE
        super().add(*args, **kwargs)

    def _before_run(
        self,
        query: str,
        **kwargs: Any,
    ) -> Any:
        if "txt" in kwargs:
            self.add(kwargs["txt"])

    def _run(
        self,
        search_query: str,
        **kwargs: Any,
    ) -> Any:
        return super()._run(query=search_query, **kwargs)
