"""Tool for reading input files provided to the crew."""

from __future__ import annotations

import base64
import io
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, PrivateAttr

from crewai.tools.base_tool import BaseTool


if TYPE_CHECKING:
    from crewai_files import FileInput

logger = logging.getLogger(__name__)


class ReadFileToolSchema(BaseModel):
    """Schema for read file tool arguments."""

    file_name: str = Field(..., description="The name of the input file to read")


class ReadFileTool(BaseTool):
    """Tool for reading input files provided to the crew kickoff.

    Provides agents access to files passed via the `files` key in inputs.
    """

    name: str = "read_file"
    description: str = (
        "Read content from an input file by name. "
        "Returns file content as text for text files, "
        "extracted text for PDFs, or base64 for other binary files."
    )
    args_schema: type[BaseModel] = ReadFileToolSchema

    _files: dict[str, FileInput] | None = PrivateAttr(default=None)

    def set_files(self, files: dict[str, FileInput] | None) -> None:
        """Set available input files.

        Args:
            files: Dictionary mapping file names to file inputs.
        """
        self._files = files

    def _run(self, file_name: str, **kwargs: object) -> str:
        """Read an input file by name.

        Args:
            file_name: The name of the file to read.

        Returns:
            File content as text for text files, extracted text for PDFs,
            or base64 encoded for other binary files.
        """
        if not self._files:
            return "No input files available."

        if file_name not in self._files:
            available = ", ".join(self._files.keys())
            return f"File '{file_name}' not found. Available files: {available}"

        file_input = self._files[file_name]
        content = file_input.read()
        content_type = file_input.content_type
        filename = file_input.filename or file_name

        text_types = (
            "text/",
            "application/json",
            "application/xml",
            "application/x-yaml",
        )

        if any(content_type.startswith(t) for t in text_types):
            return content.decode("utf-8")

        if content_type == "application/pdf":
            return self._extract_pdf_text(content, filename)

        encoded = base64.b64encode(content).decode("ascii")
        return f"[Binary file: {filename} ({content_type})]\nBase64: {encoded}"

    @staticmethod
    def _extract_pdf_text(content: bytes, filename: str) -> str:
        """Extract text from PDF bytes using pypdf.

        Falls back to a short error message (never base64) when extraction
        is not possible, so that the LLM context stays small.

        Args:
            content: Raw PDF bytes.
            filename: Name of the PDF file (for logging/messages).

        Returns:
            Extracted text, or a short diagnostic message on failure.
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.warning(
                "pypdf is not installed — cannot extract text from '%s'. "
                "Install it with: pip install pypdf",
                filename,
            )
            return (
                f"[PDF file: {filename}] "
                "Unable to extract text: pypdf is not installed. "
                "Install it with: pip install pypdf"
            )

        try:
            reader = PdfReader(io.BytesIO(content))
            pages: list[str] = []
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    pages.append(f"--- Page {page_num} ---\n{page_text}")
            if pages:
                return "\n\n".join(pages)
            return (
                f"[PDF file: {filename}] "
                "No extractable text found (the PDF may contain only images)."
            )
        except Exception as exc:
            logger.warning("Failed to extract text from PDF '%s': %s", filename, exc)
            return f"[PDF file: {filename}] Failed to extract text: {exc}"
