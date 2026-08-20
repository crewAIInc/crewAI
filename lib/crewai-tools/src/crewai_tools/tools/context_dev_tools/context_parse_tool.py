from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.security.safe_path import validate_file_path
from crewai_tools.tools.context_dev_tools.base import ContextDevBaseTool, compact


MAX_PARSE_FILE_BYTES = 25 * 1024 * 1024


class ContextParseToolSchema(BaseModel):
    file_path: str = Field(
        min_length=1,
        description="Path to the local document to convert into Markdown.",
    )
    extension: str | None = Field(
        default=None,
        description="Optional file extension hint such as pdf, docx, xlsx, or pptx.",
    )
    include_links: bool = Field(
        default=True,
        description="Preserve hyperlinks in the Markdown output.",
    )
    include_images: bool = Field(
        default=False,
        description="Include image references in the Markdown output.",
    )
    use_main_content_only: bool = Field(
        default=False,
        description="Extract only the main content from HTML-like files.",
    )
    ocr: bool = Field(
        default=False,
        description="Run OCR on selected PDF pages without usable text layers.",
    )
    pdf_start: int | None = Field(
        default=None,
        ge=1,
        description="First 1-based PDF page to parse.",
    )
    pdf_end: int | None = Field(
        default=None,
        ge=1,
        description="Last 1-based PDF page to parse.",
    )


class ContextParseTool(ContextDevBaseTool):
    name: str = "Context.dev document parser"
    description: str = (
        "Convert a local PDF, Office document, spreadsheet, presentation, image, "
        "code file, or text file into clean Markdown with Context.dev."
    )
    args_schema: type[BaseModel] = ContextParseToolSchema
    base_dir: str | None = Field(
        default=None,
        exclude=True,
        description="Directory that runtime file paths must stay inside.",
    )

    def _run(
        self,
        file_path: str,
        extension: str | None = None,
        include_links: bool = True,
        include_images: bool = False,
        use_main_content_only: bool = False,
        ocr: bool = False,
        pdf_start: int | None = None,
        pdf_end: int | None = None,
    ) -> Any:
        safe_file_path = validate_file_path(file_path, self.base_dir)
        file_size = os.path.getsize(safe_file_path)
        if file_size > MAX_PARSE_FILE_BYTES:
            raise ValueError(
                "Context.dev document parsing supports files up to 25 MiB."
            )
        if pdf_start is not None and pdf_end is not None and pdf_end < pdf_start:
            raise ValueError("pdf_end must be greater than or equal to pdf_start.")

        pdf_options = compact({"start": pdf_start, "end": pdf_end})
        return self._request(
            "POST",
            "/parse",
            params=compact(
                {
                    "extension": extension
                    or Path(safe_file_path).suffix.lstrip(".")
                    or None,
                    "includeLinks": include_links,
                    "includeImages": include_images,
                    "useMainContentOnly": use_main_content_only,
                    "ocr": ocr,
                    "pdf": pdf_options or None,
                }
            ),
            content=Path(safe_file_path).read_bytes(),
        )
