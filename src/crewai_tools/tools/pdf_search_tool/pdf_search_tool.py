from typing import Any, Optional, Type

from embedchain.models.data_type import DataType
from pydantic import BaseModel, Field, model_validator

from ..rag.rag_tool import RagTool


class FixedPDFSearchToolSchema(BaseModel):
    """Input for PDFSearchTool."""

    query: str = Field(
        ..., description="Mandatory query you want to use to search the PDF's content"
    )


class PDFSearchToolSchema(FixedPDFSearchToolSchema):
    """Input for PDFSearchTool."""

    pdf: str = Field(..., description="Mandatory pdf path you want to search")


class PDFSearchTool(RagTool):
    name: str = "Search a PDF's content"
    description: str = (
        "A tool that can be used to semantic search a query from a PDF's content."
    )
    args_schema: Type[BaseModel] = PDFSearchToolSchema

    def __init__(self, pdf: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if pdf is not None:
            self.add(pdf)
            self.description = f"A tool that can be used to semantic search a query the {pdf} PDF's content."
            self.args_schema = FixedPDFSearchToolSchema
            self._generate_description()

    def add(self, pdf: str) -> None:
        super().add(pdf, data_type=DataType.PDF_FILE)

    def _run(
        self,
        query: str,
        pdf: Optional[str] = None,
    ) -> str:
        if pdf is not None:
            self.add(pdf)
        return super()._run(query=query)
