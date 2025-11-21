from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from crewai_tools.rag.data_types import DataType
from crewai_tools.tools.rag.rag_tool import RagTool


class FixedPDFSearchToolSchema(BaseModel):
    """Input for PDFSearchTool."""

    query: str = Field(
        ..., description="Mandatory query you want to use to search the PDF's content"
    )


class PDFSearchToolSchema(FixedPDFSearchToolSchema):
    """Input for PDFSearchTool."""

    pdf: str = Field(..., description="File path or URL of a PDF file to be searched")


class PDFSearchTool(RagTool):
    name: str = "Search a PDF's content"
    description: str = (
        "A tool that can be used to semantic search a query from a PDF's content."
    )
    args_schema: type[BaseModel] = PDFSearchToolSchema
    pdf: str | None = None

    @model_validator(mode="after")
    def _configure_for_pdf(self) -> Self:
        """Configure tool for specific PDF if provided."""
        if self.pdf is not None:
            self.add(self.pdf)
            self.description = f"A tool that can be used to semantic search a query the {self.pdf} PDF's content."
            self.args_schema = FixedPDFSearchToolSchema
            self._generate_description()
        return self

    def add(self, pdf: str) -> None:
        super().add(pdf, data_type=DataType.PDF_FILE)

    def _run(  # type: ignore[override]
        self,
        query: str,
        pdf: str | None = None,
        similarity_threshold: float | None = None,
        limit: int | None = None,
    ) -> str:
        if pdf is not None:
            self.add(pdf)
        return super()._run(
            query=query, similarity_threshold=similarity_threshold, limit=limit
        )
