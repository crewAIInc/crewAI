from typing import Optional, Type

try:
    from embedchain.models.data_type import DataType
    EMBEDCHAIN_AVAILABLE = True
except ImportError:
    EMBEDCHAIN_AVAILABLE = False

from pydantic import BaseModel, Field

from ..rag.rag_tool import RagTool


class FixedCSVSearchToolSchema(BaseModel):
    """Input for CSVSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the CSV's content",
    )


class CSVSearchToolSchema(FixedCSVSearchToolSchema):
    """Input for CSVSearchTool."""

    csv: str = Field(..., description="File path or URL of a CSV file to be searched")


class CSVSearchTool(RagTool):
    name: str = "Search a CSV's content"
    description: str = (
        "A tool that can be used to semantic search a query from a CSV's content."
    )
    args_schema: Type[BaseModel] = CSVSearchToolSchema

    def __init__(self, csv: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if csv is not None:
            self.add(csv)
            self.description = f"A tool that can be used to semantic search a query the {csv} CSV's content."
            self.args_schema = FixedCSVSearchToolSchema
            self._generate_description()

    def add(self, csv: str) -> None:
        if not EMBEDCHAIN_AVAILABLE:
            raise ImportError("embedchain is not installed. Please install it with `pip install crewai-tools[embedchain]`")
        super().add(csv, data_type=DataType.CSV)

    def _run(
        self,
        search_query: str,
        csv: Optional[str] = None,
    ) -> str:
        if csv is not None:
            self.add(csv)
        return super()._run(query=search_query)
