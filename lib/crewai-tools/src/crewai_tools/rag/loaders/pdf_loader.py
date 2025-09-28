"""PDF loader for extracting text from PDF files."""

import os
from pathlib import Path
from typing import Any

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent


class PDFLoader(BaseLoader):
    """Loader for PDF files."""
    
    def load(self, source: SourceContent, **kwargs) -> LoaderResult:
        """Load and extract text from a PDF file.
        
        Args:
            source: The source content containing the PDF file path
            
        Returns:
            LoaderResult with extracted text content
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ImportError: If required PDF libraries aren't installed
        """
        try:
            import pypdf
        except ImportError:
            try:
                import PyPDF2 as pypdf
            except ImportError:
                raise ImportError(
                    "PDF support requires pypdf or PyPDF2. "
                    "Install with: uv add pypdf"
                )
        
        file_path = source.source
        
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        text_content = []
        metadata: dict[str, Any] = {
            "source": str(file_path),
            "file_name": Path(file_path).name,
            "file_type": "pdf"
        }
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                metadata["num_pages"] = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"Page {page_num}:\n{page_text}")
        except Exception as e:
            raise ValueError(f"Error reading PDF file {file_path}: {str(e)}")
        
        if not text_content:
            content = f"[PDF file with no extractable text: {Path(file_path).name}]"
        else:
            content = "\n\n".join(text_content)
        
        return LoaderResult(
            content=content,
            source=str(file_path),
            metadata=metadata,
            doc_id=self.generate_doc_id(source_ref=str(file_path), content=content)
        )