from pydantic import BaseModel, Field

from crewai_tools.tools.rag.rag_tool import RagTool


class FixedXMLSearchToolSchema(BaseModel):
    """Input for XMLSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the XML's content",
    )


class XMLSearchToolSchema(FixedXMLSearchToolSchema):
    """Input for XMLSearchTool."""

    xml: str = Field(..., description="File path or URL of a XML file to be searched")


class XMLSearchTool(RagTool):
    name: str = "Search a XML's content"
    description: str = (
        "A tool that can be used to semantic search a query from a XML's content."
    )
    args_schema: type[BaseModel] = XMLSearchToolSchema

    def __init__(self, xml: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if xml is not None:
            self.add(xml)
            self.description = f"A tool that can be used to semantic search a query the {xml} XML's content."
            self.args_schema = FixedXMLSearchToolSchema
            self._generate_description()

    def _run(  # type: ignore[override]
        self,
        search_query: str,
        xml: str | None = None,
        similarity_threshold: float | None = None,
        limit: int | None = None,
    ) -> str:
        if xml is not None:
            self.add(xml)
        return super()._run(
            query=search_query, similarity_threshold=similarity_threshold, limit=limit
        )
