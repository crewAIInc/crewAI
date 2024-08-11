from typing import Any, Optional, Type
from pydantic import BaseModel, Field
from pypdf import PdfReader, PdfWriter, PageObject, ContentStream, NameObject, Font
from pathlib import Path


class PDFTextWritingToolSchema(BaseModel):
    """Input schema for PDFTextWritingTool."""
    pdf_path: str = Field(..., description="Path to the PDF file to modify")
    text: str = Field(..., description="Text to add to the PDF")
    position: tuple = Field(..., description="Tuple of (x, y) coordinates for text placement")
    font_size: int = Field(default=12, description="Font size of the text")
    font_color: str = Field(default="0 0 0 rg", description="RGB color code for the text")
    font_name: Optional[str] = Field(default="F1", description="Font name for standard fonts")
    font_file: Optional[str] = Field(None, description="Path to a .ttf font file for custom font usage")
    page_number: int = Field(default=0, description="Page number to add text to")


class PDFTextWritingTool(RagTool):
    """A tool to add text to specific positions in a PDF, with custom font support."""
    name: str = "PDF Text Writing Tool"
    description: str = "A tool that can write text to a specific position in a PDF document, with optional custom font embedding."
    args_schema: Type[BaseModel] = PDFTextWritingToolSchema

    def run(self, pdf_path: str, text: str, position: tuple, font_size: int, font_color: str,
            font_name: str = "F1", font_file: Optional[str] = None, page_number: int = 0, **kwargs) -> str:
        reader = PdfReader(pdf_path)
        writer = PdfWriter()

        if page_number >= len(reader.pages):
            return "Page number out of range."

        page: PageObject = reader.pages[page_number]
        content = ContentStream(page["/Contents"].data, reader)

        if font_file:
            # Check if the font file exists
            if not Path(font_file).exists():
                return "Font file does not exist."

            # Embed the custom font
            font_name = self.embed_font(writer, font_file)

        # Prepare text operation with the custom or standard font
        x_position, y_position = position
        text_operation = f"BT /{font_name} {font_size} Tf {x_position} {y_position} Td ({text}) Tj ET"
        content.operations.append([font_color])  # Set color
        content.operations.append([text_operation])  # Add text

        # Replace old content with new content
        page[NameObject("/Contents")] = content
        writer.add_page(page)

        # Save the new PDF
        output_pdf_path = "modified_output.pdf"
        with open(output_pdf_path, "wb") as out_file:
            writer.write(out_file)

        return f"Text added to {output_pdf_path} successfully."

    def embed_font(self, writer: PdfWriter, font_file: str) -> str:
        """Embeds a TTF font into the PDF and returns the font name."""
        with open(font_file, "rb") as file:
            font = Font.true_type(file.read())
        font_ref = writer.add_object(font)
        return font_ref